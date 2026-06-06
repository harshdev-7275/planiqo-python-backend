from typing import Any

from langchain.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field

from chains.sanitize import wrap_untrusted
from clients.embeddings import embedding_client as _default_embedding_client
from graph.queries import (
    find_similar_issues,
    find_user_by_name,
    hybrid_similar_issues,
    suggest_assignee,
)
from graph.sync import upsert_issue


async def _safe_embed(client: Any, text: str) -> list[float] | None:
    """Embed ``text`` for the vector index, returning ``None`` on any failure.

    A missing embedding is not fatal: ``upsert_issue`` simply skips the
    vector field and ``hybrid_similar_issues`` falls back to its word-overlap
    pre-filter. Per AIService.md, an embedding-provider outage must degrade
    the smart features, never break the user-facing write."""
    try:
        vector: list[float] = await client.embed(text)
        return vector
    except Exception as e:
        logger.warning("embedding failed, proceeding without vector: {}", e)
        return None


class _NoInput(BaseModel):
    pass


class GetStatusesTool(BaseTool):
    name: str = "get_statuses"
    description: str = "List workflow statuses for the project. Returns id and name of each status. No args needed."
    args_schema: type[BaseModel] = _NoInput

    api: Any
    org_slug: str
    project_id: str

    async def _arun(self) -> str:
        logger.info("tool=get_statuses org={} project={}", self.org_slug, self.project_id)
        try:
            statuses: list[dict[str, Any]] = await self.api.get(
                f"/bot/orgs/{self.org_slug}/projects/{self.project_id}/statuses"
            )
            if not statuses:
                return "No statuses found."
            lines = [f"{s['id']} — {s['name']}{' (default)' if s.get('isDefault') else ''}" for s in statuses]
            return "\n".join(lines)
        except Exception as e:
            logger.error("tool=get_statuses failed: {}", e)
            return f"Failed to get statuses: {e}"

    def _run(self) -> str:
        raise NotImplementedError("Use async")


class GetIssuesTool(BaseTool):
    name: str = "get_issues"
    description: str = "List all issues in the current project. No args needed."
    args_schema: type[BaseModel] = _NoInput

    api: Any
    org_slug: str
    project_id: str

    @staticmethod
    def _format_issue_line(i: dict[str, Any]) -> str:
        """One issue → '#N [priority] title — Status · Assignee'.

        Status and assignee come from the bot list's resolved objects
        (``status``/``assignee``); both degrade gracefully when absent so the
        agent always gets a usable line. Showing the assignee is what lets the
        agent answer "who is assigned to what" — without it the agent truthfully
        but unhelpfully reports it has no assignee data."""
        status = i.get("status") or {}
        status_name = status.get("name") or "No status"
        assignee = i.get("assignee")
        who = assignee.get("name") if isinstance(assignee, dict) else None
        who = who or "Unassigned"
        return f"#{i['number']} [{i['priority']}] {i['title']} — {status_name} · {who}"

    async def _arun(self) -> str:
        logger.info("tool=get_issues org={} project={}", self.org_slug, self.project_id)
        try:
            issues: list[dict[str, Any]] = await self.api.get_issues(self.org_slug, self.project_id)
            if not issues:
                return "No issues found."
            lines = [self._format_issue_line(i) for i in issues]
            logger.info("tool=get_issues returned {} issues", len(issues))
            # Issue titles are user-authored — wrap so the agent treats them as
            # data, not instructions (item 19).
            return wrap_untrusted("\n".join(lines))
        except Exception as e:
            logger.error("tool=get_issues failed: {}", e)
            return f"Failed to get issues: {e}"

    def _run(self) -> str:
        raise NotImplementedError("Use async")


class _CreateIssueInput(BaseModel):
    title: str = Field(description="Issue title")
    type: str = Field(default="task", description="task|bug|story|epic")
    priority: str = Field(default="medium", description="low|medium|high|critical")
    assignee_id: str | None = Field(default=None, description="Assignee user ID")
    status_id: str | None = Field(default=None, description="Status ID — omit to use project default")


class CreateIssueTool(BaseTool):
    name: str = "create_issue"
    description: str = "Create an issue. Args: title, type(task/bug/story/epic), priority(low/medium/high/critical), assignee_id?, status_id?."
    args_schema: type[BaseModel] = _CreateIssueInput

    api: Any
    org_slug: str
    project_id: str
    user_id: str | None = None
    neo4j_client: Any | None = None
    embedding_client: Any | None = None

    async def _arun(
        self,
        title: str,
        type: str = "task",
        priority: str = "medium",
        assignee_id: str | None = None,
        status_id: str | None = None,
    ) -> str:
        logger.info("tool=create_issue title='{}' type={} priority={}", title, type, priority)
        try:
            resolved_status_id = status_id
            if not resolved_status_id:
                default = await self.api.get_default_status(self.org_slug, self.project_id)
                if not default:
                    return "Failed to create issue: no statuses configured for this project"
                resolved_status_id = default["id"]
                logger.debug("tool=create_issue default status='{}' id={}", default["name"], resolved_status_id)

            body: dict[str, Any] = {
                "title": title,
                "type": type,
                "priority": priority,
                "statusId": resolved_status_id,
            }
            if assignee_id:
                body["assigneeId"] = assignee_id

            result = await self.api.post(
                f"/orgs/{self.org_slug}/projects/{self.project_id}/issues",
                body,
                user_id=self.user_id,
            )
            logger.info("tool=create_issue created #{}", result.get("number"))
            response = f"Created #{result['number']}: {result['title']}"

            # Incremental Neo4j sync (Sprint 2.1): the issue was just
            # created in Postgres — mirror it to the graph so smart
            # features (find_similar_issues, suggest_assignee) see it
            # immediately, not on the next nightly full_sync. Failures
            # are LOGGED and swallowed: a graph outage must not break the
            # user-facing response.
            #
            # The Node POST response typically contains only number/title
            # — the validated type/priority/assigneeId we already have on
            # `body`, so we merge them into the upsert payload.
            if self.neo4j_client:
                # Compute the embedding so the vector index (Sprint 2.2) has
                # data to rank on — without this the index is empty and
                # hybrid_similar_issues silently falls back to word-overlap
                # for every query. Embedding failures degrade to no-vector.
                embedding = await _safe_embed(
                    self.embedding_client or _default_embedding_client, title
                )
                try:
                    await upsert_issue(
                        self.neo4j_client, {**body, **result}, embedding=embedding
                    )
                except Exception as e:
                    logger.error(
                        "tool=create_issue neo4j upsert failed (issue was "
                        "still created in Postgres; smart features will "
                        "see it on the next full_sync): {}",
                        e, exc_info=True,
                    )

            if not self.neo4j_client:
                logger.debug("tool=create_issue suggestion skipped: no neo4j_client injected")
            elif assignee_id:
                logger.debug("tool=create_issue suggestion skipped: explicit assignee_id={}", assignee_id)
            else:
                try:
                    suggestions = await suggest_assignee(
                        self.neo4j_client, self.project_id, title, type
                    )
                    logger.debug(
                        "tool=create_issue suggest_assignee returned {} candidates",
                        len(suggestions),
                    )
                    if not suggestions:
                        logger.info(
                            "tool=create_issue no suggestion: empty result — "
                            "no project members in graph, run POST /graph/sync"
                        )
                    else:
                        top = suggestions[0]
                        reason = (
                            f"based on past {type} issues"
                            if top["score"] > 0
                            else "most available team member"
                        )
                        response += (
                            f"\n\U0001f4a1 Suggested assignee: {top['userName']} ({reason})"
                        )
                        logger.info(
                            "tool=create_issue suggested user='{}' score={:.1f} reason='{}'",
                            top["userName"], top["score"], reason,
                        )
                except Exception as e:
                    logger.error(
                        "tool=create_issue suggestion failed (neo4j error): {} — "
                        "issue was still created successfully",
                        e, exc_info=True,
                    )

            return response
        except Exception as e:
            logger.error("tool=create_issue failed: {}", e)
            return f"Failed to create issue: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class _SuggestAssigneeInput(BaseModel):
    title: str = Field(description="Issue title")
    type: str = Field(default="task", description="task|bug|story|epic")


class SuggestAssigneeTool(BaseTool):
    name: str = "suggest_assignee"
    description: str = "Suggest the best person to assign a new issue. Args: title, type(task/bug/story/epic)."
    args_schema: type[BaseModel] = _SuggestAssigneeInput

    neo4j_client: Any
    org_slug: str
    project_id: str

    async def _arun(self, title: str, type: str = "task") -> str:
        logger.info("tool=suggest_assignee title='{}' type={}", title, type)
        try:
            suggestions = await suggest_assignee(self.neo4j_client, self.project_id, title, type)
            if not suggestions:
                return "No suggestion available — no project members found in graph (run /graph/sync)."
            top = suggestions[0]
            if top["score"] > 0:
                return (
                    f"Suggested assignee: {top['userName']}"
                    f" (score: {top['score']:.0f}, based on past {type} issues)"
                )
            return (
                f"Suggested assignee: {top['userName']}"
                f" (most available team member)"
            )
        except Exception as e:
            logger.error("tool=suggest_assignee failed: {}", e)
            return f"Failed to suggest assignee: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class _FindSimilarInput(BaseModel):
    title: str = Field(description="Issue title to search for similar issues")


class FindSimilarIssuesTool(BaseTool):
    name: str = "find_similar_issues"
    description: str = "Find issues similar to a given title. Args: title."
    args_schema: type[BaseModel] = _FindSimilarInput

    neo4j_client: Any
    org_slug: str
    project_id: str
    embedding_client: Any | None = None

    async def _arun(self, title: str) -> str:
        logger.info("tool=find_similar_issues title='{}'", title)
        try:
            # Sprint 2.2: prefer the hybrid (word-overlap pre-filter + vector
            # kNN rerank) path so "login fails" matches "can't sign in". If the
            # embedding can't be computed (provider outage), fall back to the
            # pure word-overlap query — hybrid itself also degrades on a
            # vector-index error.
            embedding = await _safe_embed(
                self.embedding_client or _default_embedding_client, title
            )
            if embedding is not None:
                similar = await hybrid_similar_issues(
                    self.neo4j_client, self.project_id, title, embedding
                )
            else:
                similar = await find_similar_issues(
                    self.neo4j_client, self.project_id, title
                )
            if not similar:
                return "No similar issues found."
            lines = [f"#{s['number']}: {s['title']}" for s in similar]
            # Titles are user-authored — wrap as data (item 19).
            return "Similar issues found:\n" + wrap_untrusted("\n".join(lines))
        except Exception as e:
            logger.error("tool=find_similar_issues failed: {}", e)
            return f"Failed to find similar issues: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class _FindUserInput(BaseModel):
    name: str = Field(description="Person's name or partial name (e.g. 'Alice', 'ali')")


class FindUserByNameTool(BaseTool):
    """Resolve a person the user mentions by name to a project member record.

    MUST be called BEFORE create_issue whenever the user says something like
    "assign to Alice" or "give it to Harsh" — the create_issue tool expects a
    user ID, not a display name. Up to 5 matches are returned; pick the one
    whose name/email best matches what the user said, or ask the user to
    disambiguate if there are several.
    """

    name: str = "find_user_by_name"
    description: str = (
        "Find a project member by name. Args: name. Returns up to 5 matching "
        "members with their id, name, and email. Use this to resolve a name "
        "into a user ID before calling create_issue with an assignee."
    )
    args_schema: type[BaseModel] = _FindUserInput

    neo4j_client: Any
    org_slug: str
    project_id: str

    async def _arun(self, name: str) -> str:
        logger.info("tool=find_user_by_name name='{}'", name)
        try:
            results = await find_user_by_name(self.neo4j_client, self.project_id, name)
            if not results:
                return f"No project member matching '{name}'."
            lines = [f"id={r['id']} name={r['name']} email={r['email']}" for r in results]
            return "\n".join(lines)
        except Exception as e:
            logger.error("tool=find_user_by_name failed: {}", e)
            return f"Failed to find user: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class _UpdateIssueInput(BaseModel):
    issue_number: int = Field(description="Issue number the user referred to, e.g. 5 for '#5'")
    priority: str | None = Field(default=None, description="low|medium|high|critical")
    title: str | None = Field(default=None, description="New title")
    assignee_id: str | None = Field(
        default=None,
        description="Assignee user ID — resolve a name with find_user_by_name first",
    )


class UpdateIssueTool(BaseTool):
    """Update fields (priority, title, assignee) on an existing issue.

    The user refers to an issue by its NUMBER ("#5"), but the backend keys on
    the issue's UUID. This tool resolves the number to an ID itself, so the
    model only needs the number it already has — it never has to surface a raw
    UUID. To change status use update_issue_status; to move an issue into a
    sprint use add_issue_to_sprint.
    """

    name: str = "update_issue"
    description: str = (
        "Update an existing issue's priority, title, or assignee. "
        "Args: issue_number (e.g. 5), priority?, title?, assignee_id?."
    )
    args_schema: type[BaseModel] = _UpdateIssueInput

    api: Any
    org_slug: str
    project_id: str
    user_id: str | None = None
    neo4j_client: Any | None = None
    embedding_client: Any | None = None

    async def _arun(
        self,
        issue_number: int,
        priority: str | None = None,
        title: str | None = None,
        assignee_id: str | None = None,
    ) -> str:
        logger.info("tool=update_issue number={} project={}", issue_number, self.project_id)
        body: dict[str, Any] = {}
        if priority:
            body["priority"] = priority
        if title:
            body["title"] = title
        if assignee_id:
            body["assigneeId"] = assignee_id
        if not body:
            return "Nothing to update — specify a priority, title, or assignee."
        try:
            issues: list[dict[str, Any]] = await self.api.get_issues(self.org_slug, self.project_id)
            match = next((i for i in issues if i.get("number") == issue_number), None)
            if not match:
                return f"Issue #{issue_number} not found in this project."
            await self.api.patch(
                f"/orgs/{self.org_slug}/projects/{self.project_id}/issues/{match['id']}",
                body,
                user_id=self.user_id,
            )
            logger.info("tool=update_issue updated #{} fields={}", issue_number, list(body))
            # Incremental Neo4j sync (Sprint 2.1): mirror the updated
            # issue to the graph so smart features see the new
            # priority/title/assignee immediately. Failures are
            # LOGGED and swallowed — a graph outage must not break the
            # user-facing response.
            #
            # Merge the pre-update row with the patch body so the upsert
            # reflects the new state (Node's PATCH response is often just
            # the touched fields, not the whole row).
            if self.neo4j_client:
                # Re-embed on the effective (possibly new) title so the vector
                # index tracks the current text. Failures degrade to no-vector.
                merged = {**match, **body}
                embedding = await _safe_embed(
                    self.embedding_client or _default_embedding_client,
                    str(merged.get("title") or ""),
                )
                try:
                    await upsert_issue(self.neo4j_client, merged, embedding=embedding)
                except Exception as e:
                    logger.error(
                        "tool=update_issue neo4j upsert failed (issue was "
                        "still updated in Postgres; smart features will "
                        "see it on the next full_sync): {}",
                        e, exc_info=True,
                    )
            return f"Updated #{issue_number}."
        except Exception as e:
            logger.error("tool=update_issue failed: {}", e)
            return f"Failed to update issue: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")


class _UpdateIssueStatusInput(BaseModel):
    issue_id: str = Field(description="Issue ID")
    status_id: str = Field(description="New status ID")


class UpdateIssueStatusTool(BaseTool):
    name: str = "update_issue_status"
    description: str = "Update an issue's status. Args: issue_id, status_id."
    args_schema: type[BaseModel] = _UpdateIssueStatusInput

    api: Any
    org_slug: str
    project_id: str
    user_id: str | None = None

    async def _arun(self, issue_id: str, status_id: str) -> str:
        logger.info("tool=update_issue_status issue_id={} status_id={}", issue_id, status_id)
        try:
            await self.api.patch(
                f"/orgs/{self.org_slug}/projects/{self.project_id}/issues/{issue_id}/status",
                {"statusId": status_id},
                user_id=self.user_id,
            )
            logger.info("tool=update_issue_status updated {}", issue_id)
            return f"Updated issue {issue_id} status."
        except Exception as e:
            logger.error("tool=update_issue_status failed: {}", e)
            return f"Failed to update issue: {e}"

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")
