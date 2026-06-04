import pytest
from unittest.mock import AsyncMock, MagicMock

from tools.member_tools import ListMembersTool, MemberIssuesTool

CTX = {"org_slug": "acme", "project_id": "proj-1"}


def _mock_api(**kwargs):
    api = MagicMock()
    for method, mock in kwargs.items():
        setattr(api, method, mock)
    return api


_MEMBERS = [
    {"userId": "u1", "name": "Alice Smith", "email": "alice@acme.com", "role": "admin"},
    {"userId": "u2", "name": "Bob Jones", "email": "bob@acme.com", "role": "member"},
]

_ISSUES = [
    {"number": 5, "title": "Login broken", "priority": "high", "assigneeId": "u1",
     "status": {"name": "Todo"}},
    {"number": 6, "title": "Dark mode", "priority": "low", "assigneeId": "u2",
     "status": {"name": "In Progress"}},
    {"number": 7, "title": "Checkout flow", "priority": "medium", "assigneeId": "u1",
     "status": {"name": "Todo"}},
]


# --- ListMembersTool ---

@pytest.mark.asyncio
async def test_list_members_formats_name_role_email():
    api = _mock_api(get_project_members=AsyncMock(return_value=_MEMBERS))
    tool = ListMembersTool(api=api, **CTX)
    result = await tool._arun()
    assert "Alice Smith" in result
    assert "admin" in result
    assert "Bob Jones" in result
    assert "member" in result


@pytest.mark.asyncio
async def test_list_members_empty_message():
    api = _mock_api(get_project_members=AsyncMock(return_value=[]))
    tool = ListMembersTool(api=api, **CTX)
    result = await tool._arun()
    assert "No" in result and "member" in result.lower()


@pytest.mark.asyncio
async def test_list_members_calls_api_with_context():
    api = _mock_api(get_project_members=AsyncMock(return_value=[]))
    tool = ListMembersTool(api=api, org_slug="acme", project_id="proj-42")
    await tool._arun()
    api.get_project_members.assert_called_once_with("acme", "proj-42")


@pytest.mark.asyncio
async def test_list_members_error_string():
    api = _mock_api(get_project_members=AsyncMock(side_effect=Exception("timeout")))
    tool = ListMembersTool(api=api, **CTX)
    result = await tool._arun()
    assert "Failed" in result


# --- MemberIssuesTool ---

@pytest.mark.asyncio
async def test_member_issues_lists_only_that_members_issues():
    api = _mock_api(
        get_project_members=AsyncMock(return_value=_MEMBERS),
        get_issues=AsyncMock(return_value=_ISSUES),
    )
    tool = MemberIssuesTool(api=api, **CTX)
    result = await tool._arun(name="Alice")
    assert "#5" in result          # assigned to u1 (Alice)
    assert "#7" in result          # assigned to u1 (Alice)
    assert "#6" not in result      # assigned to u2 (Bob) — must be excluded


@pytest.mark.asyncio
async def test_member_issues_resolves_by_email_too():
    api = _mock_api(
        get_project_members=AsyncMock(return_value=_MEMBERS),
        get_issues=AsyncMock(return_value=_ISSUES),
    )
    tool = MemberIssuesTool(api=api, **CTX)
    result = await tool._arun(name="bob@acme.com")
    assert "#6" in result


@pytest.mark.asyncio
async def test_member_issues_name_not_found_lists_members():
    api = _mock_api(
        get_project_members=AsyncMock(return_value=_MEMBERS),
        get_issues=AsyncMock(return_value=_ISSUES),
    )
    tool = MemberIssuesTool(api=api, **CTX)
    result = await tool._arun(name="Zoe")
    assert "Zoe" in result and "no team member" in result.lower()
    assert "Alice Smith" in result  # lists who IS on the team


@pytest.mark.asyncio
async def test_member_issues_ambiguous_asks_to_disambiguate():
    members = [
        {"userId": "u1", "name": "Alice Smith", "email": "alice@acme.com", "role": "admin"},
        {"userId": "u3", "name": "Alicia Jones", "email": "alicia@acme.com", "role": "member"},
    ]
    api = _mock_api(
        get_project_members=AsyncMock(return_value=members),
        get_issues=AsyncMock(return_value=_ISSUES),
    )
    tool = MemberIssuesTool(api=api, **CTX)
    result = await tool._arun(name="Ali")
    assert "Alice Smith" in result and "Alicia Jones" in result
    assert "specific" in result.lower() or "which" in result.lower()


@pytest.mark.asyncio
async def test_member_issues_member_with_no_issues():
    api = _mock_api(
        get_project_members=AsyncMock(return_value=_MEMBERS),
        get_issues=AsyncMock(return_value=[]),
    )
    tool = MemberIssuesTool(api=api, **CTX)
    result = await tool._arun(name="Alice")
    assert "Alice Smith" in result and "no" in result.lower()


@pytest.mark.asyncio
async def test_member_issues_error_string():
    api = _mock_api(get_project_members=AsyncMock(side_effect=Exception("503")))
    tool = MemberIssuesTool(api=api, **CTX)
    result = await tool._arun(name="Alice")
    assert "Failed" in result
