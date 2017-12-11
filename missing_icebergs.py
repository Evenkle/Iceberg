from dataset import read_dataset


def missing_icebergs():
    original_ids = [it['id'] for it in read_dataset('test.json')]
    processed_ids = []
    for file in [
        'test_processed_0',
        'test_processed_1',
        'test_processed_2',
        'test_processed_3',
        'test_processed_4',
        'test_processed_5',
        'test_processed_6',
        'test_processed_7',
        'test_processed_8',
        'test_processed_9',
    ]:
        processed_ids += [it['id'] for it in read_dataset(file + '.json')]
        print(len(processed_ids), len(original_ids))
    for i in range(len(original_ids)):
        print(i)
        if original_ids[i] != processed_ids[i]:
            print('Not equal!', original_ids[i], processed_ids[i], i)

missing_icebergs()