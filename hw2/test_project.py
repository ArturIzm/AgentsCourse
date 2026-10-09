import asyncio
import json
import random

import pytest
from mcp import Client
from server import mcp
from experiment import LoopGuard, has_loop, failure_response, make_loop_tasks, unstable_search


def call_tool(name, arguments):
    async def run():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(run())


@pytest.mark.parametrize(
    "name, arguments, expected_ids",
    [
        (
            "schedule_get",
            {"lesson_id": "lesson_1"},
            ["lesson_1"],
        ),
        (
            "schedule_search",
            {"date": "2026-10-10"},
            ["lesson_1", "lesson_2"],
        ),
        (
            "schedule_teacher",
            {"teacher": "А. Иванова"},
            ["lesson_1", "lesson_3", "lesson_4"],
        ),
        (
            "schedule_room",
            {"room": "204"},
            ["lesson_1", "lesson_3"],
        ),
    ],
)
@pytest.mark.parametrize("response_format", ["concise", "detailed"])
def test_success(name, arguments, expected_ids, response_format):
    arguments = {**arguments, "response_format": response_format}
    result = call_tool(name, arguments)

    assert result.is_error is False

    lessons = [json.loads(item.text) for item in result.content]
    assert [lesson["id"] for lesson in lessons] == expected_ids

    for lesson in lessons:
        if response_format == "detailed":
            assert "teacher" in lesson
            assert "type" in lesson
        else:
            assert "teacher" not in lesson
            assert "type" not in lesson


@pytest.mark.parametrize(
    "name, arguments, hint",
    [
        (
            "schedule_get",
            {"lesson_id": "lesson_999"},
            "schedule_search",
        ),
        (
            "schedule_search",
            {"date": "2020-01-01"},
            "другую дату",
        ),
        (
            "schedule_teacher",
            {"teacher": "Неизвестный"},
            "Укажите имя",
        ),
        (
            "schedule_room",
            {"room": "999"},
            "Укажите номер",
        ),
    ],
)
def test_error(name, arguments, hint):
    result = call_tool(name, arguments)

    assert result.is_error is True
    assert hint in result.content[0].text


    
def test_extra_argument():
    result = call_tool(
        "schedule_get",
        {"lesson_id": "lesson_1", "banana": True},
    )

    assert result.is_error is True
    assert "banana" in result.content[0].text


def test_tool_schemas():
    async def run():
        async with Client(mcp) as client:
            return await client.list_tools()

    result = asyncio.run(run())

    assert len(result.tools) == 4

    for tool in result.tools:
        schema = tool.input_schema
        assert schema["additionalProperties"] is False

        fields = schema["properties"]
        assert fields["response_format"]["enum"] == ["concise", "detailed"]

        for field in fields.values():
            assert field["description"]



def test_guard_repeat_in_window():
    guard = LoopGuard()
    assert guard.check('A', {'x': 1}, '1') is None
    assert guard.check('B', {'x': 2}, '2') is None
    assert guard.check('A', {'x': 1}, '3') is None
    assert guard.check('C', {}, '4') is None
    assert guard.check('A', {'x': 1}, '5').startswith('повтор')


def test_guard_oscillation():
    guard = LoopGuard()
    for name in ['A', 'B', 'A']:
        assert guard.check(name, {}, name) is None
    assert guard.check('B', {}, 'другое наблюдение').startswith('колебание')


def test_guard_no_progress():
    guard = LoopGuard(max_same=100)
    for i in range(5):
        assert guard.check('A', {'x': i}, 'empty') is None
    assert guard.check('A', {'x': 5}, 'empty').startswith('нет прогресса')


def test_guard_normalizes_argument_order():
    guard = LoopGuard()
    guard.check('A', {'x': 1, 'y': 2}, '1')
    guard.check('A', {'y': 2, 'x': 1}, '2')
    assert guard.check('A', {'x': 1, 'y': 2}, '3').startswith('повтор')


def test_guard_window():
    guard = LoopGuard()
    guard.check('A', {}, '0')
    for i in range(6):
        guard.check('B', {'x': i}, str(i))
    assert guard.check('A', {}, 'new') is None


def test_loop_metric():
    assert has_loop(['A', 'A', 'A'])
    assert has_loop(['A', 'B', 'A', 'B'])
    assert not has_loop(['A', 'B', 'A', 'C', 'A'])
    assert not has_loop(['A', 'B', 'C'])


def test_empty_and_hint():
    empty = failure_response()
    hint = failure_response(helpful=True)
    assert empty.content == [] and empty.is_error is False
    assert hint.is_error is True
    assert 'date' in hint.content[0].text


def test_unstable_search():
    async def run():
        async with Client(mcp, mode='2026-07-28') as client:
            args = {'date': '2026-10-10', 'response_format': 'concise'}
            empty, draw1 = await unstable_search(client, args, random.Random(1))
            result, draw2 = await unstable_search(client, args, random.Random(2))
            assert draw1 < 0.5 and empty.content == []
            assert draw2 >= 0.5 and result.is_error is False
            assert json.loads(result.content[0].text)['id'] == 'lesson_1'
    asyncio.run(run())


def test_loop_tasks():
    tasks = make_loop_tasks()
    assert len(tasks) == 20
    assert len({task['text'] for task in tasks}) == 20
    assert all(task['expected_ids'] for task in tasks)
