"""Tests for GraphSyncService."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from ai_service.graph.sync import (
    GraphSyncService,
    SyncStats,
    _build_change_rels,
    _build_comment_rels,
)
from ai_service.neo4j import Neo4jClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingSession:
    """Captures every Cypher query + parameters issued during a sync."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, query: str, **kwargs: Any) -> MagicMock:
        self.calls.append((query, kwargs))
        result = MagicMock()
        result.data = AsyncMock(return_value=[])
        return result

    def queries(self) -> list[str]:
        return [q for q, _ in self.calls]

    def params_for(self, fragment: str) -> dict[str, Any] | None:
        for q, params in self.calls:
            if fragment in q:
                return params
        return None


def _make_neo4j(session: _RecordingSession) -> Neo4jClient:
    client = MagicMock(spec=Neo4jClient)

    @asynccontextmanager
    async def _session() -> AsyncIterator[_RecordingSession]:
        yield session

    client.session = _session  # type: ignore[method-assign]
    return client  # type: ignore[return-value]


def _make_backend(
    *,
    projects: list[dict[str, Any]] | None = None,
    categories: list[dict[str, Any]] | None = None,
    sprints: list[dict[str, Any]] | None = None,
    issues: list[dict[str, Any]] | None = None,
    members: list[dict[str, Any]] | None = None,
    comments: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> MagicMock:
    backend = MagicMock()
    backend.list_projects = AsyncMock(return_value=projects or [])
    backend.list_categories = AsyncMock(return_value=categories or [])
    backend.list_sprints = AsyncMock(return_value=sprints or [])
    backend.list_issues = AsyncMock(return_value=issues or [])
    backend.list_org_members = AsyncMock(return_value=members or [])
    backend.list_issue_comments = AsyncMock(return_value=comments or [])
    backend.list_issue_history = AsyncMock(return_value=history or [])
    return backend


_PROJECT = {"id": "11111111-1111-1111-1111-111111111111", "name": "Alpha"}
_ORG_SLUG = "acme"
_ORG_ID = "00000000-0000-0000-0000-000000000001"
_CAT_ID = "22222222-2222-2222-2222-222222222222"
_SPRINT_ID = "33333333-3333-3333-3333-333333333333"
_USER_ID = "44444444-4444-4444-4444-444444444444"
_ISSUE_ID = "55555555-5555-5555-5555-555555555555"


def _issue(
    *,
    issue_id: str = _ISSUE_ID,
    assignee_id: str | None = None,
    parent_id: str | None = None,
    completed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "id": issue_id,
        "number": 1,
        "title": "Test issue",
        "description": None,
        "type": "bug",
        "priority": "medium",
        "statusId": "st-1",
        "categoryId": _CAT_ID,
        "sprintId": None,
        "assigneeId": assignee_id,
        "parentId": parent_id,
        "completedAt": completed_at,
        "createdAt": "2024-01-01T00:00:00Z",
    }


class TestSyncProjectNodes:
    async def test_upserts_org_node(self) -> None:
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend())
        await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert any("MERGE (o:Org" in q for q in sess.queries())

    async def test_upserts_project_node_with_belongs_to(self) -> None:
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend())
        await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert any("MERGE (p:Project" in q and "BELONGS_TO" in q for q in sess.queries())

    async def test_upserts_categories(self) -> None:
        cats = [{"id": _CAT_ID, "name": "Frontend", "color": "#f00", "projectId": _PROJECT["id"]}]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(categories=cats))
        stats = await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert stats.categories == 1
        assert any("IN_PROJECT" in q for q in sess.queries())

    async def test_upserts_sprints(self) -> None:
        sprints = [
            {"id": _SPRINT_ID, "name": "Sprint 1", "status": "active",
             "startDate": None, "endDate": None, "projectId": _PROJECT["id"]}
        ]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(sprints=sprints))
        stats = await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert stats.sprints == 1

    async def test_upserts_members(self) -> None:
        # node-backend's member list has TWO ids: `id` = membership row id,
        # `userId` = the actual user id that issues/history/comments reference.
        members = [{"id": "membership-row-1", "userId": _USER_ID,
                    "name": "Alice", "email": "alice@acme.com"}]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(members=members))
        stats = await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert stats.users == 1

    async def test_member_user_node_keyed_on_user_id_not_membership_id(self) -> None:
        # Regression: keying the User node on the membership `id` creates a
        # node that never matches assignee/changedBy/comment refs (which use
        # userId), leaving behavioral rels orphaned. Must MERGE on m.userId.
        members = [{"id": "membership-row-1", "userId": _USER_ID,
                    "name": "Alice", "email": "alice@acme.com"}]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(members=members))
        await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        member_q = next(q for q in sess.queries() if "MEMBER_OF" in q)
        assert "MERGE (u:User {id: m.userId})" in member_q
        assert "m.id})" not in member_q

    async def test_upserts_issues(self) -> None:
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(issues=[_issue()]))
        stats = await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert stats.issues == 1


class TestSyncBehavioralRelationships:
    async def test_creates_resolved_rel_for_completed_issue(self) -> None:
        issues = [_issue(assignee_id=_USER_ID, completed_at="2024-06-01T10:00:00Z")]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(issues=issues))
        await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert any("RESOLVED" in q for q in sess.queries())

    async def test_no_resolved_rel_when_not_completed(self) -> None:
        issues = [_issue(assignee_id=_USER_ID, completed_at=None)]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(issues=issues))
        await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert not any("RESOLVED" in q for q in sess.queries())

    async def test_creates_commented_on_rel(self) -> None:
        comments = [
            {"id": "c-1", "issueId": _ISSUE_ID, "body": "Looks good",
             "author": {"id": _USER_ID, "name": "Bob"}, "createdAt": "2024-01-02T00:00:00Z",
             "updatedAt": "2024-01-02T00:00:00Z", "isEdited": False, "parentId": None}
        ]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(issues=[_issue()], comments=comments))
        await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        commented = next(q for q in sess.queries() if "COMMENTED_ON" in q)
        # The Issue must be MATCHed (it already exists with a uniqueness
        # constraint); inlining (i:Issue {id}) in the rel MERGE would try to
        # CREATE a duplicate node and raise ConstraintError.
        assert "MATCH (i:Issue {id: r.issue_id})" in commented
        assert "COMMENTED_ON]->(i)" in commented
        assert "COMMENTED_ON]->(i:Issue" not in commented

    async def test_creates_changed_rel_from_history(self) -> None:
        history = [
            {"id": "h-1", "issueId": _ISSUE_ID, "fieldChanged": "title",
             "oldValue": "Old", "newValue": "New",
             "changedAt": "2024-01-03T00:00:00Z",
             "changedBy": {"id": _USER_ID, "name": "Carol", "avatarUrl": None}}
        ]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(issues=[_issue()], history=history))
        await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        changed = next(q for q in sess.queries() if "CHANGED" in q)
        # Same constraint as COMMENTED_ON: MATCH the Issue, never inline-create it.
        assert "MATCH (i:Issue {id: r.issue_id})" in changed
        assert "CHANGED]->(i)" in changed
        assert "CHANGED]->(i:Issue" not in changed

    async def test_subtask_of_rel_created_when_parent_id_set(self) -> None:
        parent_id = "66666666-6666-6666-6666-666666666666"
        child_id = "77777777-7777-7777-7777-777777777777"
        issues = [
            _issue(issue_id=parent_id),
            _issue(issue_id=child_id, parent_id=parent_id),
        ]
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), _make_backend(issues=issues))
        await svc.sync_project(_ORG_SLUG, _ORG_ID, _PROJECT)
        assert any("SUBTASK_OF" in q for q in sess.queries())


class TestFullSync:
    async def test_calls_sync_project_for_each_project(self) -> None:
        projects = [
            {"id": "11111111-1111-1111-1111-111111111111", "name": "Alpha"},
            {"id": "22222222-2222-2222-2222-222222222222", "name": "Beta"},
        ]
        sess = _RecordingSession()
        backend = _make_backend(projects=projects)
        svc = GraphSyncService(_make_neo4j(sess), backend)
        stats = await svc.full_sync(_ORG_SLUG, _ORG_ID)
        assert backend.list_categories.call_count == 2
        assert stats.projects == 2

    async def test_aggregates_stats_across_projects(self) -> None:
        projects = [
            {"id": "11111111-1111-1111-1111-111111111111", "name": "Alpha"},
            {"id": "22222222-2222-2222-2222-222222222222", "name": "Beta"},
        ]
        backend = _make_backend(projects=projects, issues=[_issue()])
        sess = _RecordingSession()
        svc = GraphSyncService(_make_neo4j(sess), backend)
        stats = await svc.full_sync(_ORG_SLUG, _ORG_ID)
        assert stats.issues == 2  # 1 issue x 2 projects


class TestSyncStats:
    def test_merge_adds_counts(self) -> None:
        a = SyncStats(projects=1, issues=5, relationships=10)
        b = SyncStats(projects=1, issues=3, relationships=6, errors=["e1"])
        merged = a.merge(b)
        assert merged.projects == 2
        assert merged.issues == 8
        assert merged.relationships == 16
        assert merged.errors == ["e1"]

    def test_to_dict_has_all_keys(self) -> None:
        s = SyncStats(projects=1, categories=2, sprints=3, issues=4, users=5, relationships=6)
        d = s.to_dict()
        assert set(d.keys()) == {"projects", "categories", "sprints", "issues", "users", "relationships", "errors"}


class TestBuildCommentRels:
    def test_groups_by_author(self) -> None:
        signals = {
            "iss-1": [
                {"author": {"id": "u-1", "name": "Alice"}, "createdAt": "2024-01-01T00:00:00Z"},
                {"author": {"id": "u-1", "name": "Alice"}, "createdAt": "2024-01-02T00:00:00Z"},
                {"author": {"id": "u-2", "name": "Bob"}, "createdAt": "2024-01-01T00:00:00Z"},
            ]
        }
        rels = _build_comment_rels(signals)
        by_user = {r["user_id"]: r for r in rels}
        assert by_user["u-1"]["count"] == 2
        assert by_user["u-2"]["count"] == 1

    def test_skips_entries_without_author_id(self) -> None:
        signals = {"iss-1": [{"author": {}, "createdAt": "2024-01-01T00:00:00Z"}]}
        rels = _build_comment_rels(signals)
        assert rels == []


class TestBuildChangeRels:
    def test_groups_by_changer(self) -> None:
        signals = {
            "iss-1": [
                {"changedBy": {"id": "u-1", "name": "Alice"}, "changedAt": "2024-01-01T00:00:00Z",
                 "fieldChanged": "title", "oldValue": "A", "newValue": "B"},
                {"changedBy": {"id": "u-1", "name": "Alice"}, "changedAt": "2024-01-02T00:00:00Z",
                 "fieldChanged": "priority", "oldValue": "low", "newValue": "high"},
            ]
        }
        rels = _build_change_rels(signals)
        assert len(rels) == 1
        assert rels[0]["count"] == 2
