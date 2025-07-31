from utils.utils import *

def load_data(path, seed=None):
    if seed is not None:
        random.seed(seed)

    data = read_file(path)
    data['consistency'] = 1

    def change_opinion_order(test_item):
        lines = test_item.split('\n')
        option_a, option_b = lines[-2], lines[-1]
        option_a = option_a[2:].strip()
        option_b = option_b[2:].strip()
        lines[-2] = f'A. {option_b}'
        lines[-1] = f'B. {option_a}'
        new_test_item = '\n'.join(lines)
        return new_test_item

    # shuffle word question
    for ridx, row in data.iterrows():
        if random.random() < 0.5:
            data.loc[ridx, 'speak'] = change_opinion_order(row['speak'])
            answer = row['correct_answer']
            if answer == 'A':
                answer = 'B'
            elif answer == 'B':
                answer = 'A'
            data.loc[ridx, 'correct_answer'] = answer
            data.loc[ridx, 'consistency'] = 1 - row['consistency']

    # shuffle deed question
    for ridx, row in data.iterrows():
        if random.random() < 0.5:
            data.loc[ridx, 'act'] = change_opinion_order(row['act'])
            data.loc[ridx, 'consistency'] = 1 - row['consistency']

    return data