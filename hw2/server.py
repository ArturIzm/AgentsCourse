from mcp.server import MCPServer
from typing import Annotated, Literal
from pydantic import Field
from mcp.types import CallToolResult, TextContent


ResponseFormat = Literal["concise", "detailed"]

mcp = MCPServer("schedule")

LESSONS = [
    {
        "id": "lesson_1",
        "subject": "Автономные агенты",
        "date": "2026-10-10",
        "time": "12:00",
        "room": "204",
        "teacher": "А. Иванова",
        "type": "lecture",
    },
    {
        "id": "lesson_2",
        "subject": "Python",
        "date": "2026-10-10",
        "time": "14:00",
        "room": "305",
        "teacher": "И. Петров",
        "type": "practice",
    },
    {
        "id": "lesson_3",
        "subject": "Машинное обучение",
        "date": "2026-10-12",
        "time": "10:00",
        "room": "204",
        "teacher": "А. Иванова",
        "type": "lecture",
    },
    {
        "id": "lesson_4",
        "subject": "Автономные агенты",
        "date": "2026-10-12",
        "time": "12:00",
        "room": "410",
        "teacher": "А. Иванова",
        "type": "practice",
    },
    {
        "id": "lesson_5",
        "subject": "Python",
        "date": "2026-10-13",
        "time": "10:00",
        "room": "305",
        "teacher": "И. Петров",
        "type": "lecture",
    },
    {
        "id": "lesson_6",
        "subject": "Машинное обучение",
        "date": "2026-10-13",
        "time": "14:00",
        "room": "410",
        "teacher": "М. Соколова",
        "type": "practice",
    },
]


def tool_error(message):
    return CallToolResult(
        is_error=True,
        content=[
            TextContent(type="text", text=message)
        ]
    )

def format_lessons(lessons, response_format):
    if response_format == "detailed":
        return [lesson.copy() for lesson in lessons]

    return [
        {
            "id": lesson["id"],
            "subject": lesson["subject"],
            "date": lesson["date"],
            "time": lesson["time"],
            "room": lesson["room"],
        }
        for lesson in lessons
    ]


@mcp.tool()
def schedule_get(
    lesson_id: Annotated[
        str,
        Field(description="ID занятия, полученный через schedule_search."),
    ],
    response_format: Annotated[
        ResponseFormat,
        Field(description="concise — основные поля; detailed — полная карточка."),
    ] = "concise",
):
    """Получить занятие по ID. Не используйте для поиска: для этого есть schedule_search."""

    if response_format not in ["concise", "detailed"]:
        return tool_error(
            "response_format должен быть concise или detailed."
        )

    for lesson in LESSONS:
        if lesson["id"] == lesson_id:
            if response_format == "concise":
                return {
                    "id": lesson["id"],
                    "subject": lesson["subject"],
                    "date": lesson["date"],
                    "time": lesson["time"],
                    "room": lesson["room"],
                }

            return lesson.copy()

    return tool_error(
        f"Занятие {lesson_id} не найдено. "
        "Найдите существующий ID через schedule_search."
    )



@mcp.tool()
def schedule_search(
    date: Annotated[
        str,
        Field(description="Дата занятий в формате YYYY-MM-DD, например 2026-10-10."),
    ],
    response_format: Annotated[
        ResponseFormat,
        Field(description="concise — основные поля; detailed — полные карточки."),
    ] = "concise",
):
    """Найти занятия на дату. Не используйте для получения карточки по известному ID."""

    if response_format not in ["concise", "detailed"]:
        return tool_error(
            "response_format должен быть concise или detailed."
        )   

    found = []
    for lesson in LESSONS:
        if lesson["date"] == date:
            found.append(lesson)

    if not found:
        return tool_error(
            "На эту дату занятий нет. "
            "Попробуйте другую дату в формате YYYY-MM-DD."
        )

    return format_lessons(found, response_format)

@mcp.tool()
def schedule_teacher(
    teacher: Annotated[
        str,
        Field(description="Имя преподавателя как в расписании, например А. Иванова."),
    ],
    response_format: Annotated[
        ResponseFormat,
        Field(description="concise — основные поля; detailed — полные карточки."),
    ] = "concise",
):
    """Найти занятия преподавателя. Не используйте для поиска по аудитории или ID."""

    if response_format not in ["concise", "detailed"]:
        return tool_error(
            "response_format должен быть concise или detailed."
        )

    found = []
    for lesson in LESSONS:
        if lesson["teacher"].casefold() == teacher.strip().casefold():
            found.append(lesson)

    if not found:
        return tool_error(
            "Преподаватель не найден. "
            "Укажите имя как в карточке занятия, например А. Иванова."
        )

    return format_lessons(found, response_format)

@mcp.tool()
def schedule_room(
    room: Annotated[
        str,
        Field(description="Номер аудитории как в расписании, например 204."),
    ],
    response_format: Annotated[
        ResponseFormat,
        Field(description="concise — основные поля; detailed — полные карточки."),
    ] = "concise",
):
    """Найти занятия в аудитории. Не используйте для поиска по преподавателю или ID."""

    if response_format not in ["concise", "detailed"]:
        return tool_error(
        "response_format должен быть concise или detailed."
    )

    found = []
    for lesson in LESSONS:
        if lesson["room"] == room.strip():
            found.append(lesson)

    if not found:
        return tool_error(
            "Аудитория не найдена. "
            "Укажите номер как в карточке занятия, например 204."
        )

    return format_lessons(found, response_format)


for tool in mcp._tool_manager.list_tools():
    args_model = tool.fn_metadata.arg_model
    args_model.model_config["extra"] = "forbid"
    args_model.model_rebuild(force=True)
    tool.parameters = args_model.model_json_schema(by_alias=True)

if __name__ == "__main__":
    mcp.run()