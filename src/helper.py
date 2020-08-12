import itertools
import tensorflow as tf


def prepare_lm_inputs_labels(
        text_tensor,
        vectorize_layer
        ):
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    Eg. text_tensor(shape=(None, 1), dtype=string) = [
            'foo bar code review',
        ]
        #maxlen = 3
        => tokenized_sentences(shape=(None, 101), dtype=int64) = [
            [1, 2, 3, 4],
        ]
        =>
        x(shape=(None, 101), dtype=int64) = [1, 2, 3]
        y(shape=(None, 101), dtype=int64) = [2, 3, 4]
    """
    text_tensor = tf.expand_dims(text_tensor, -1)
    tokenized_sentences = vectorize_layer(text_tensor)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


def show_dataset(ds, num_items=5):
    print('Spec:', ds.element_spec)
    data = list(itertools.islice(ds.as_numpy_iterator(), num_items))
    print(data)
    return data
