# AI Assistant — Manual UI Test Catalog

Scenarios to exercise the conversational AI (the **AI Assistant** chat page) end to end.
Type each message into the chat box and check the result against **Expect**. Tick the box
when it behaves as described.

> These are **manual / exploratory** tests against the live stack (frontend → Node `/api/chat`
> → AI `/chat` → Groq + Neo4j + Node API). The automated unit suite is separate:
> `uv run pytest -m "not integration"`.

---

## 0. Prerequisites (read first — skipping these will give misleading results)

- [ ] **Node backend + AI service + frontend all running**, and a provider key set (`AI_PROVIDER=groq`, `GROQ_API_KEY`).
- [ ] **You are inside a real project** in the UI. Writes validate against `project_id`; with no
      project context, pre-flight validation is silently skipped and a confirmed write can fail at execution.
- [ ] **Graph is synced:** `POST /graph/sync` for the org has been run. Assignee resolution and
      smart-assignee read **Neo4j** — with an empty graph, `assign to <Member>` wrongly fails
      validation ("No team member named…") even though the member exists in Postgres.
- [ ] **Your user has `member` role** on the project (writes impersonate you via `X-Bot-User-Id`).
- [ ] Replace placeholders below: `<Member>` = a real project member, `#N` = a real issue number,
      `Sprint X` = a real sprint name.

**Status badges to watch in the UI:** `awaiting_confirmation` (shows Yes/No buttons) ·
`executed` · `cancelled` · `validation_failed` · `quota_exceeded`.

---

## 1. 🔥 Complex multi-feature scenarios (highest value)

Each one stresses several subsystems at once.

- [ ] **C1 — everything at once.** Type:
      `Create a critical bug for the checkout page and assign it to <Member> in Sprint 1`
      **Expect:** type=bug, priority=**critical** (the "critical" keyword beats bug's default "high"),
      title cleaned to `checkout page` (leading "Create a" and trailing "in Sprint 1" stripped),
      assignee + sprint pre-validated, then a 3-clause preview:
      *"I'll create a bug titled 'checkout page' with critical priority, assign it to <Member>, and
      put it in sprint Sprint 1. Reply 'yes' to confirm or 'no' to cancel."* → reply `yes` → executes.

- [ ] **C2 — title scan-from-end.** Type: `Add a bug for checkout to Bob`
      **Expect:** title becomes `bug for checkout` (only the *rightmost* prepositional phrase "to Bob"
      is stripped — NOT "for checkout to Bob"); assignee = Bob (validated).

- [ ] **C3 — urgency keyword + colon title.** Type:
      `open a ticket: users can't reset their password, it's urgent`
      **Expect:** priority=**critical** ("urgent"); reasonable title; preview → confirm.

- [ ] **C4 — multi-field update by number + name.** Type:
      `change #N to high priority and reassign it to <Member>`
      **Expect:** resolves `#N`→UUID internally and `<Member>`→user id, preview
      *"I'll update #N set priority to high, and set assignee to <Member>."* → `yes`.

- [ ] **C5 — validation short-circuits the preview.** Type: `create a bug, assign to Zaphod`
      **Expect:** NO preview — immediate *"No team member named 'Zaphod' found in this project.
      Team members: …"* (status `validation_failed`).

---

## 2. CREATE_ISSUE — conventions / title-cleaning / validation

**Convention normalizer** (priority & type inferred from words; explicit value always wins):

- [ ] `file a bug: search returns no results` → type=**bug**, priority=**high** (bug default)
- [ ] `prod is down, create an incident` → priority=**critical** (multi-word phrase)
- [ ] `this is a p0 — payments are failing` → priority=**critical**
- [ ] `add a blocker for the release` → priority=**critical**
- [ ] `create a task to update the docs, it's important` → priority=**high**
- [ ] `make a low priority task to clean up logs` → priority=**low** (explicit beats keywords)

**Title cleaning:**

- [ ] `Create a login page to <Member>` → title `login page` (verb+determiner + trailing "to <Member>" stripped)
- [ ] `Login page` → title stays `Login page` (nothing to strip)

**Validation failures (should NOT reach a preview):**

- [ ] `create a task in Sprint 99` → *"No sprint matching 'Sprint 99'… Available sprints: …"*
- [ ] `create a bug assigned to Nobody` → *"No team member named 'Nobody'… Team members: …"*

**Sprint fuzzy-match (should *pass* if "Sprint 1" exists):**

- [ ] `create a task in sprin1` → resolves to "Sprint 1" (prefix-stripped match)
- [ ] `create a task in Sprint 99` → still fails (suffix differs from "Sprint 1")

---

## 3. Write-confirmation flow (the gate)

Send a create first (e.g. `create a bug for the navbar`), then reply:

- [ ] `yes` / `yep` / `do it` / `yeah go ahead` / `sounds good` → executes (`executed`)
- [ ] `no` / `nope` / `cancel` / `never mind` → *"Okay, cancelled. Anything else?"* (`cancelled`)
- [ ] `actually that's the wrong one` → **contextual negation** → cancelled (catches "wrong")
- [ ] `none of those` → contextual negation → cancelled
- [ ] `ok to delete production` → must **NOT** confirm (not a clean affirmation) → drops pending & reclassifies
- [ ] send an unrelated `show me all sprints` while a proposal is pending → stale proposal dropped, new intent runs
- [ ] wait **10+ minutes**, then `yes` → pending **expired** (600s TTL) → treated as a fresh message, not a confirmation

---

## 4. UPDATE_ISSUE — priority / title / reassign

- [ ] `set #N priority to critical` → preview *"I'll update #N set priority to critical."* → `yes`
- [ ] `rename issue N to "Login is completely broken"` → title update
- [ ] `reassign #N to <Member>` → name + number resolved
- [ ] `move #N to Done` → status path (uses `update_issue_status`, a different tool)
- [ ] `change #99999 to high priority` → **validation:** *"Issue #99999 not found in this project."* (no preview)

---

## 5. CREATE_SPRINT

- [ ] `create a sprint called Q3 Planning` → *"I'll create sprint 'Q3 Planning'. Reply 'yes'…"* → `yes` → created
- [ ] `start a new sprint named Sprint 7 with the goal ship the checkout redesign` → name + goal sent
- [ ] After `yes`, confirm the sprint actually appears in the sprint list (this path previously confirmed then did nothing).

---

## 6. QUERY_MEMBER (runs immediately, no confirmation)

- [ ] `who's on this team?` → member list with roles
- [ ] `what is <Member> working on?` → only that member's assigned issues
- [ ] `show me <Member>'s issues` → same
- [ ] `what is <Member> working on?` for a member with no issues → *"<Member> has no assigned issues."*
- [ ] `what is Zaphod working on?` → *"No team member named 'Zaphod'…"* + lists the real team
- [ ] `what is Al working on?` when "Alice" **and** "Alicia" exist → *"Multiple members match 'Al': … Please be more specific…"*

---

## 7. SUMMARIZE (deterministic sprint summary — no LLM, so output is stable)

- [ ] `summarize Sprint 2` → header + `Progress: D of N done.` + `Status: Todo 2, In Progress 1, Done 3` + `Priority: critical 1, high 2…`
- [ ] `give me a summary of the current sprint` → falls back to the **active** sprint
- [ ] `summarize sprint 99` → *"I couldn't find a sprint to summarize…"*
- [ ] `summarize the sprint` with no active sprint → same not-found guidance

---

## 8. Reads (immediate)

- [ ] `show me all issues`
- [ ] `what's in the backlog`
- [ ] `list all sprints`
- [ ] `what's the active sprint`
- [ ] `show high priority issues`

---

## 9. Multi-turn / memory (windowed history per `user_id:org:project`)

Run each as a **sequence** in one chat:

- [ ] `show me all issues` → `which of those are high priority?` (2nd turn uses context)
- [ ] `create a critical bug for the API timeout` → `yes` → `now summarize the active sprint`
- [ ] `reassign #N to <Member>` → `no` → `actually set its priority to high` (fresh pending after the cancel)

---

## 10. Edge / stubs / adversarial

- [ ] `what's the weather today?` → *"I didn't understand that. Try asking about issues, sprints, or team members."* (UNKNOWN)
- [ ] team-channel-ish phrasing (e.g. `give me the teams context`) → likely *"teams_agent not yet implemented"* (TEAMS_CONTEXT is the one remaining stub)
- [ ] empty message / `asdfqwer` → UNKNOWN help
- [ ] `create a bug 🐛 with émojis & "quotes" for the <script>alert(1)</script> page` → title cleans, renders safely, no injection
- [ ] `create 5 bugs for login, signup, checkout, search, and profile` → **known limitation:** one issue per turn (won't batch-create) — confirm expected behavior

---

## Two behaviors worth watching closely

- [ ] **Smart-assignee suggestion:** after a `create` with **no** assignee, with a synced graph you
      should see an extra line *"💡 Suggested assignee: <name> (based on past bug issues)"*. Empty graph → no suggestion.
- [ ] **Quota guard (optional):** set `ORG_TOKEN_QUOTA` to a low number, exceed it, and confirm the
      reply is *"This workspace has reached its AI usage limit…"* (status `quota_exceeded`) with no LLM call.

---

## Intent coverage map

| Intent | Implemented | Section |
|---|---|---|
| CREATE_ISSUE | ✅ | 1, 2, 3 |
| UPDATE_ISSUE | ✅ | 1, 4 |
| QUERY_ISSUES | ✅ | 8, 9 |
| QUERY_SPRINT | ✅ | 8 |
| CREATE_SPRINT | ✅ | 5 |
| QUERY_MEMBER | ✅ | 6 |
| SUMMARIZE | ✅ | 7 |
| TEAMS_CONTEXT | ❌ stub (behavior undefined) | 10 |
| UNKNOWN | ✅ | 10 |
