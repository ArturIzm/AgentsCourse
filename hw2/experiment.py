import argparse
import asyncio
import json
import random
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mcp import Client
from mcp.types import CallToolResult, TextContent
from server import LESSONS, mcp

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / 'data.json'
MODEL = 'unsloth/qwen3.5-4b'
BASE_URL = 'http://localhost:1234'
TEMPERATURE = 0.3
MAX_TOKENS = 256
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
MAX_STEPS = 10
CONDITIONS = ['empty', 'guard', 'hint']


def read_data():
    return json.loads(DATA_FILE.read_text())


def save_data(data):
    temporary = DATA_FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False) + '\n')
    temporary.replace(DATA_FILE)


def save_part(name, value):
    data = read_data()
    data[name] = value
    save_data(data)


def append_log(name, row):
    data = read_data()
    data.setdefault(name, []).append(row)
    save_data(data)


def read_logs(name):
    return read_data().get(name, [])

def request_model(messages):
    payload = {
        'model': MODEL,
        'messages': messages,
        'temperature': TEMPERATURE,
        'max_tokens': MAX_TOKENS,
        'stream': False,
    }
    request = Request(
        BASE_URL + '/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    row = {'payload': payload, 'raw_response': None, 'error': None}
    started = time.monotonic()
    try:
        with urlopen(request, timeout=60) as response:
            row['raw_response'] = json.load(response)
    except HTTPError as error:
        row['error'] = f'HTTP {error.code}: {error.read().decode("utf-8", errors="replace")}'
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    row['seconds'] = time.monotonic() - started
    return row


def parse_response(call):
    if call['error']:
        raise ValueError(call['error'])
    choice = call['raw_response']['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise ValueError('Ответ не завершён: лимит выходных токенов')
    content = choice['message']['content']
    answer = json.loads(content)
    if not isinstance(answer, dict):
        raise ValueError('Ожидался JSON-объект')
    return answer


def tool_text(tool):
    return f'{tool["name"]}: {tool["description"]}'


def make_messages(text, tools):
    descriptions = '\n'.join(tool_text(tool) for tool in tools)
    return [
        {'role': 'system', 'content': (
            'Выбери один инструмент для запроса пользователя. '
            'Используй только инструменты из списка. '
            'Верни только JSON вида {"tool": "имя_инструмента"}.'
        )},
        {'role': 'user', 'content': f'Инструменты:\n{descriptions}\n\nЗапрос: {text}'},
    ]


def load_data():
    return read_data()['dataset']


def prepare_retrieval(data):
    saved = read_data().get('retrieval')
    if saved:
        return saved
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL, device='cpu', local_files_only=True)
    texts = [q['text'] for q in data['queries']]
    query_vectors = model.encode(texts, normalize_embeddings=True)
    retrieval = {}
    for n, tools in data['tool_sets'].items():
        tool_vectors = model.encode([tool_text(t) for t in tools], normalize_embeddings=True)
        scores = query_vectors @ tool_vectors.T
        retrieval[n] = {}
        for i, query in enumerate(data['queries']):
            indices = np.argsort(-scores[i])[:5]
            retrieval[n][query['id']] = {
                'names': [tools[int(j)]['name'] for j in indices],
                'scores': [float(scores[i, int(j)]) for j in indices],
            }
    save_part('retrieval', retrieval)
    return retrieval


def selection_attempt(query, tools, retrieved, n, variant, repeat):
    call = request_model(make_messages(query['text'], tools))
    row = {
        'key': f'{n}:{variant}:{query["id"]}:{repeat}',
        'query_id': query['id'], 'query': query['text'],
        'correct_tool': query['correct_tool'], 'N': int(n),
        'variant': variant, 'repeat': repeat,
        'candidates': [tool['name'] for tool in tools],
        'retrieved': retrieved['names'], 'retrieval_scores': retrieved['scores'],
        'retrieval_hit': query['correct_tool'] in retrieved['names'],
        'selected_tool': None, 'correct': False,
        **call,
    }
    usage = (call['raw_response'] or {}).get('usage', {})
    row['input_tokens'] = usage.get('prompt_tokens')
    row['output_tokens'] = usage.get('completion_tokens')
    try:
        answer = parse_response(call)
        if set(answer) != {'tool'}:
            raise ValueError('Ожидалось ровно поле tool')
        selected = answer['tool']
        if not isinstance(selected, str) or selected not in row['candidates']:
            raise ValueError('Инструмент отсутствует в переданном списке')
        row['selected_tool'] = selected
        row['correct'] = selected == query['correct_tool']
    except Exception as error:
        row['error'] = f'{type(error).__name__}: {error}'
    return row


def run_selection():
    data = load_data()
    assert len(data['pool']) == 100 and len(data['queries']) == 30
    retrieval = prepare_retrieval(data)
    jobs = []
    for repeat in range(3):
        for query in data['queries']:
            for n, tools in data['tool_sets'].items():
                found = retrieval[n][query['id']]
                by_name = {tool['name']: tool for tool in tools}
                for variant in ['A', 'B']:
                    candidates = tools if variant == 'A' else [by_name[name] for name in found['names']]
                    jobs.append((query, candidates, found, n, variant, repeat))
    random.Random(123).shuffle(jobs)
    path = 'selection'
    existing = read_logs(path)
    done = {row['key'] for row in existing}
    assert len(done) == len(existing)
    for job in jobs:
        query, tools, found, n, variant, repeat = job
        key = f'{n}:{variant}:{query["id"]}:{repeat}'
        if key in done:
            continue
        row = selection_attempt(*job)
        append_log(path, row)
        done.add(key)
        if len(done) % 20 == 0 or row['error']:
            print(f'{len(done)}/720; {key}; correct={row["correct"]}; error={row["error"]}', flush=True)
    assert len(done) == 720
    print('Часть A завершена: 720 попыток', flush=True)


def save_protocol():
    if 'protocol' in read_data():
        return
    with urlopen(BASE_URL + '/api/v0/models', timeout=10) as response:
        available = json.load(response)
    model_info = next(m for m in available['data'] if m['id'] == MODEL)
    save_part('protocol', {
        'model': model_info,
        'temperature': TEMPERATURE, 'max_output_tokens': MAX_TOKENS,
        'request_timeout_seconds': 60,
        'embedding_model': EMBEDDING_MODEL,
        'selection': {
            'N': [5, 20, 50, 100], 'queries': 30, 'repeats': 3,
            'attempts': 720, 'top_k': 5, 'seed_dataset': 42,
            'seed_run_order': 123, 'bootstrap_unit': 'query',
            'bootstrap_samples': 10000,
            'retrieval': 'external deterministic search, one model call per attempt',
            'token_scope': 'usage.prompt_tokens for the complete LLM request; embedding cost excluded',
        },
        'failures': 'Ошибки и незавершённые ответы считаются неправильными; автоматических повторов нет.',
        'pilot': 'selection_pilot.jsonl не входит в основные метрики',
        'budget': 'Локальная модель, платных API нет. 720 попыток выбора + 60 эпизодов петель, до 10 шагов каждый.',
    })

class LoopGuard:
    def __init__(self, max_same=3, window=6):
        self.calls, self.obs = [], []
        self.n, self.w = max_same, window

    def check(self, name, args, observation):
        self.calls.append(name + json.dumps(args, sort_keys=True))
        self.obs.append(observation)
        last = self.calls[-self.w:]
        if Counter(last)[last[-1]] >= self.n:
            return 'повтор: смените инструмент или аргументы'
        if len(last) >= 4 and last[-4:-2] == last[-2:]:
            return 'колебание между двумя действиями'
        recent = self.obs[-self.w:]
        if len(recent) == self.w and len(set(recent)) == 1:
            return 'нет прогресса: наблюдения не меняются'
        return None


def has_loop(calls):
    for i in range(len(calls)):
        last = calls[:i + 1]
        if len(last) >= 3 and last[-1] == last[-2] == last[-3]:
            return True
        if len(last) >= 4 and last[-4] == last[-2] and last[-3] == last[-1]:
            return True
    return False

def failure_response(helpful=False):
    if helpful:
        return CallToolResult(
            is_error=True,
            content=[TextContent(type='text', text='Ничего не найдено; попробуйте другое значение date.')],
        )
    return CallToolResult(content=[])


async def unstable_search(client, args, rng, helpful=False):
    draw = rng.random()
    if draw < 0.5:
        return failure_response(helpful), draw
    return await client.call_tool('schedule_search', args), draw


def make_loop_tasks():
    dates = ['2026-10-10', '2026-10-12', '2026-10-13']
    templates = [
        'Найди любое учебное занятие на одну из дат: {dates}. Получи его карточку и верни ID.',
        'Подбери одну пару на {dates}. Сначала найди её, потом открой карточку и сообщи ID.',
        'Мне нужно одно занятие в одну из дат {dates}. Найди его и проверь карточку по ID.',
        'Посмотри занятия на {dates}. Выбери любое, получи полную карточку и верни идентификатор.',
        'Узнай ID любого занятия на одну из дат {dates}. Используй поиск и получение карточки.',
    ]
    templates += [
        'Выбери занятие на {dates}, найди его поиском и открой карточку. Нужен ID.',
        'Помоги получить карточку любой пары на одну из дат {dates} через поиск и get.',
        'На {dates} хочу найти одно занятие. После поиска проверь его сведения и сообщи ID.',
        'Достань идентификатор любой пары на {dates}, проверив найденное занятие по ID.',
        'Для учебного плана найди занятие на {dates}. Получи карточку выбранного варианта.',
        'Нужна одна существующая пара на {dates}: сначала поиск, затем карточка по ID.',
        'Хочу узнать ID занятия на {dates}. Подтверди его существование поиском и карточкой.',
        'Найди и открой карточку одного занятия в одну из дат {dates}. Верни его ID.',
        'Проверь любое занятие на {dates}: найди его в расписании, потом запроси карточку.',
        'Мне подойдёт любая пара на {dates}. Найди вариант, получи подробности и верни ID.',
        'Подбери занятие из расписания на {dates} и проверь его через получение карточки.',
        'Поищи пару на {dates}, запроси её карточку и назови идентификатор.',
        'Нужен проверенный ID учебного занятия на {dates}. Используй поиск и get.',
        'Получи карточку любого найденного занятия на {dates}, затем верни ID.',
        'Найди один вариант занятия на {dates}. Убедись через карточку, что он существует.',
    ]
    tasks = []
    for i in range(20):
        chosen = [dates[i % 3], dates[(i + 1) % 3]]
        tasks.append({
            'id': f'loop_{i + 1}',
            'text': templates[i % len(templates)].format(dates=', '.join(chosen)),
            'dates': chosen,
            'expected_ids': [lesson['id'] for lesson in LESSONS if lesson['date'] in chosen],
            'seed': 300 + i,
        })
    return tasks


def starting_messages(task):
    return [
        {'role': 'system', 'content': (
            'Реши задачу с помощью инструментов. На каждом шаге верни только один JSON-объект. '
            'schedule_search(date, response_format) ищет занятия на дату YYYY-MM-DD. '
            'schedule_get(lesson_id, response_format) получает карточку занятия по известному ID. '
            'response_format: concise или detailed. '
            'Для вызова верни {"tool":"schedule_search","args":{"date":"YYYY-MM-DD","response_format":"concise"}} '
            'либо {"tool":"schedule_get","args":{"lesson_id":"ID","response_format":"detailed"}}. '
            'Для завершения верни {"final":{"lesson_id":"ID"}}. '
            'Используй ID, реально полученный из поиска, и получи его карточку перед завершением. '
            'Не считай собственное предположение результатом инструмента.'
        )},
        {'role': 'user', 'content': task['text']},
    ]


def extract_ids(result):
    if result.is_error:
        return set()
    ids = set()
    for item in result.content:
        if item.type != 'text':
            continue
        try:
            lesson = json.loads(item.text)
            if isinstance(lesson, dict) and 'id' in lesson:
                ids.add(lesson['id'])
        except (ValueError, TypeError):
            pass
    return ids


async def run_episode(client, task, condition):
    key = f'{condition}:{task["id"]}'
    rng = random.Random(task['seed'])
    guard = LoopGuard(max_same=3, window=6)
    messages = starting_messages(task)
    calls = []
    searched_ids, fetched_ids = set(), set()
    total_input = total_output = 0
    usage_complete = True
    row = {
        'key': key, 'condition': condition, 'task': task,
        'success': False, 'steps': 0, 'tool_calls': 0,
        'loop': False, 'guard_signals': 0, 'injected_empty': 0,
        'trace': [], 'stop_reason': 'step_limit',
    }
    append_log('events', {'event': 'start', 'key': key, 'task': task})

    for step in range(1, MAX_STEPS + 1):
        call = request_model(messages)
        raw = call['raw_response'] or {}
        usage = raw.get('usage', {})
        if usage.get('prompt_tokens') is None or usage.get('completion_tokens') is None:
            usage_complete = False
        total_input += usage.get('prompt_tokens') or 0
        total_output += usage.get('completion_tokens') or 0
        entry = {'step': step, **call, 'observation': None, 'guard': None}
        row['steps'] = step
        try:
            answer = parse_response(call)
            entry['action'] = answer
            content = raw['choices'][0]['message']['content']
            messages.append({'role': 'assistant', 'content': content})

            if set(answer) == {'final'}:
                final = answer['final']
                if not isinstance(final, dict) or set(final) != {'lesson_id'}:
                    raise ValueError('final должен содержать только lesson_id')
                final_id = final['lesson_id']
                row['success'] = (
                    final_id in task['expected_ids']
                    and final_id in searched_ids
                    and final_id in fetched_ids
                )
                row['final_id'] = final_id
                row['stop_reason'] = 'final' if row['success'] else 'wrong_final'
            else:
                if set(answer) != {'tool', 'args'} or not isinstance(answer['args'], dict):
                    raise ValueError('Ожидались поля tool и args')
                name, args = answer['tool'], answer['args']
                if name not in ['schedule_search', 'schedule_get']:
                    raise ValueError('Неизвестный инструмент')
                signature = name + json.dumps(args, sort_keys=True)
                calls.append(signature)
                row['tool_calls'] += 1
                row['loop'] = has_loop(calls)
                if name == 'schedule_search':
                    result, draw = await unstable_search(client, args, rng, condition == 'hint')
                    entry['random_draw'] = draw
                    entry['failure_injected'] = draw < 0.5
                    row['injected_empty'] += int(draw < 0.5)
                    searched_ids.update(extract_ids(result))
                else:
                    result = await client.call_tool(name, args)
                    fetched_ids.update(extract_ids(result))
                observation = result.model_dump(mode='json', by_alias=True, exclude_none=True)
                entry['observation'] = observation
                observation_text = json.dumps(observation, ensure_ascii=False, sort_keys=True)
                if condition == 'guard':
                    signal = guard.check(name, args, observation_text)
                    entry['guard'] = signal
                    if signal:
                        row['guard_signals'] += 1
                        observation_text += '\nLoopGuard: ' + signal
                messages.append({'role': 'user', 'content': 'Результат инструмента: ' + observation_text})
        except Exception as error:
            entry['parse_or_execution_error'] = f'{type(error).__name__}: {error}'
            row['stop_reason'] = 'error'

        row['trace'].append(entry)
        append_log('events', {'event': 'step', 'key': key, **entry})
        if row['stop_reason'] != 'step_limit':
            break

    row['input_tokens'] = total_input
    row['output_tokens'] = total_output
    row['total_tokens'] = total_input + total_output
    row['usage_complete'] = usage_complete
    return row


async def run_loops():
    tasks = make_loop_tasks()
    if 'loop_dataset' not in read_data():
        save_part('loop_dataset', {
            'tasks': tasks, 'conditions': CONDITIONS,
            'max_steps': MAX_STEPS, 'failure_probability': 0.5,
            'guard': {'max_same': 3, 'window': 6, 'source': 'Код лекции, прислан пользователем'},
            'guard_action': 'Подсказка модели; следующие действия выбирает модель, бюджет остаётся 10 шагов.',
            'metric_loop': '3 одинаковых вызова подряд или A-B-A-B независимо от наблюдений',
            'pairing': 'Одинаковый seed генератора нестабильности у каждой задачи в трёх условиях.',
            'success': 'Верный ID из разрешённых дат, реально найденный через поиск и полученный через get.',
            'runs': '20 задач в каждом из трёх условий; 3 повтора в условии ДЗ - параметр max_same=3.',
        })
    path = 'loops'
    rows = read_logs(path)
    done = {row['key'] for row in rows}
    assert len(done) == len(rows)
    started_keys = {row['key'] for row in read_logs('events') if row['event'] == 'start'}
    for key in started_keys - done:
        events = [e for e in read_logs('events') if e['key'] == key and e['event'] == 'step']
        condition, task_id = key.split(':')
        task = next(t for t in tasks if t['id'] == task_id)
        calls = [e['action']['tool'] + json.dumps(e['action']['args'], sort_keys=True)
                 for e in events if 'tool' in e.get('action', {})]
        usages = [(e.get('raw_response') or {}).get('usage', {}) for e in events]
        inputs = sum(u.get('prompt_tokens') or 0 for u in usages)
        outputs = sum(u.get('completion_tokens') or 0 for u in usages)
        append_log(path, {
            'key': key, 'condition': condition, 'task': task, 'success': False,
            'loop': has_loop(calls), 'steps': len(events), 'trace': events,
            'tool_calls': len(calls),
            'guard_signals': sum(bool(e.get('guard')) for e in events),
            'injected_empty': sum(bool(e.get('failure_injected')) for e in events),
            'input_tokens': inputs, 'output_tokens': outputs,
            'total_tokens': inputs + outputs,
            'usage_complete': False, 'stop_reason': 'interrupted',
        })
        done.add(key)
    jobs = [(task, condition) for task in tasks for condition in CONDITIONS]
    random.Random(456).shuffle(jobs)
    async with Client(mcp, mode='2026-07-28') as client:
        for task, condition in jobs:
            if f'{condition}:{task["id"]}' in done:
                continue
            row = await run_episode(client, task, condition)
            append_log(path, row)
            done.add(row['key'])
            print(f'{len(done)}/60; {row["key"]}; success={row["success"]}; loop={row["loop"]}; steps={row["steps"]}', flush=True)
    assert len(done) == 60

def selection_summary():
    rows = read_logs('selection')
    assert len(rows) == 720
    df = pd.DataFrame(rows)
    assert df['key'].nunique() == 720
    summary = []
    for (n, variant), group in df.groupby(['N', 'variant']):
        assert len(group) == 90
        assert group.groupby('query_id').size().eq(3).all()
        assert set(group['repeat']) == {0, 1, 2}
        means = group.groupby('query_id')['correct'].mean().to_numpy()
        rng = np.random.default_rng(2026)
        samples = rng.choice(means, size=(10000, len(means)), replace=True).mean(axis=1)
        low, high = np.quantile(samples, [0.025, 0.975])
        recall = group.groupby('query_id')['retrieval_hit'].first().mean()
        summary.append({
            'N': n, 'variant': variant, 'attempts': len(group),
            'correct': int(group['correct'].sum()), 'accuracy': group['correct'].mean(),
            'ci_low': low, 'ci_high': high, 'recall_at_5': recall,
            'input_tokens': group['input_tokens'].mean(),
            'token_records': int(group['input_tokens'].notna().sum()),
            'errors': int(group['error'].notna().sum()),
        })
    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    return table


def loop_summary():
    rows = read_logs('loops')
    assert len(rows) == 60
    df = pd.DataFrame(rows)
    assert df['key'].nunique() == 60
    summary = []
    for condition in ['empty', 'guard', 'hint']:
        group = df[df['condition'] == condition]
        assert len(group) == 20
        summary.append({
            'condition': condition, 'episodes': len(group),
            'loops': int(group['loop'].sum()), 'loop_rate': group['loop'].mean(),
            'successes': int(group['success'].sum()), 'success_rate': group['success'].mean(),
            'mean_steps': group['steps'].mean(), 'mean_input_tokens': group['input_tokens'].mean(),
            'mean_output_tokens': group['output_tokens'].mean(), 'mean_total_tokens': group['total_tokens'].mean(),
            'usage_complete': int(group['usage_complete'].sum()),
            'errors': int(group['stop_reason'].isin(['error', 'interrupted']).sum()),
            'step_limits': int(group['stop_reason'].eq('step_limit').sum()),
            'guard_signals': int(group['guard_signals'].sum()),
        })
    table = pd.DataFrame(summary)
    print(table.to_string(index=False))
    return table

def accuracy_plot(table):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for variant, label in [('A', 'A: all descriptions'), ('B', 'B: embedding top-5')]:
        part = table[table['variant'] == variant]
        ax.errorbar(part['N'], part['accuracy'],
                    yerr=[part['accuracy'] - part['ci_low'], part['ci_high'] - part['accuracy']],
                    marker='o', capsize=4, label=label)
    ax.set(xlabel='Number of tools N', ylabel='Selection accuracy', ylim=(-0.02, 1.05), xticks=[5, 20, 50, 100])
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    image = BytesIO()
    fig.savefig(image, format='png', dpi=180)
    plt.close(fig)
    return image.getvalue()

async def response_sizes():
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")

    examples = [
        ("schedule_get", {"lesson_id": "lesson_1"}),
        ("schedule_search", {"date": "2026-10-10"}),
        ("schedule_teacher", {"teacher": "А. Иванова"}),
    ]



    print("| Пример | concise | detailed |")
    print("|---|---:|---:|")

    async with Client(mcp) as client:
        for name, arguments in examples:
            sizes = []

            for response_format in ["concise", "detailed"]:
                result = await client.call_tool(
                    name,
                    {**arguments, "response_format": response_format},
                )

                if result.is_error:
                    raise RuntimeError(f"Не удалось вызвать {name}")

                text = "\n".join(item.text for item in result.content)
                sizes.append(len(encoding.encode(text)))

            print(f"| {name}: {arguments} | {sizes[0]} | {sizes[1]} |")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['selection', 'loops', 'analyze', 'sizes'])
    part = parser.parse_args().part
    if part == 'selection':
        save_protocol()
        run_selection()
    elif part == 'loops':
        asyncio.run(run_loops())
    elif part == 'analyze':
        selection_summary()
        loop_summary()
    elif part == 'sizes':
        asyncio.run(response_sizes())
