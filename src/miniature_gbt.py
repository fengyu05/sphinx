# miniature_gbt.py
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


class MiniatureGbtModel(object):
    def __init__(
            self,
            checkpoint_path,
            batch_size=32,
            epochs=30,
            vocab_size=10000,
            maxlen=40, #maximun token per sentence
            num_tokens_generated=40,
            n_dim=256,
            num_heads=2,  # Number of attention heads
            feed_forward_dim=256,  # Hidden layer size in feed forward network inside transformer
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.num_tokens_generated = num_tokens_generated
        self.n_dim = n_dim
        self.num_heads = num_heads
        self.feed_forward_dim = feed_forward_dim

        self.init_model(checkpoint_path=checkpoint_path)

    def init_model(self, custom_standardization_func=None, checkpoint_path=None):
        '''
        init model
        set instance property:
            self.model
            self.vectorize_layer
        '''
        self.vectorize_layer = TextVectorization(
            standardize=custom_standardization_func,
            max_tokens=self.vocab_size - 1, #this is -1 because need to keep one for 0=EOS
            output_mode="int",
            output_sequence_length=self.maxlen + 1, # this is +1 because need to get one more for the last label
        )

        inputs = layers.Input(shape=(self.maxlen,), dtype=tf.int32)
        embedding_layer = TokenAndPositionEmbedding(self.maxlen, self.vocab_size, self.n_dim)
        x = embedding_layer(inputs)
        transformer_block = TransformerBlock(self.n_dim, self.num_heads, self.feed_forward_dim)
        x = transformer_block(x)
        outputs = layers.Dense(self.vocab_size)(x)
        self.model = keras.Model(inputs=inputs, outputs=[outputs, x])

        # load weights from previous checkpoint
        if checkpoint_path:
            self.model.load_weights(checkpoint_path)
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(
            "adam", loss=[loss_fn, None],
        )  # No loss and optimization based on word embeddings from transformer block
        self.model.summary()

        return self.model

    def preprocessing(self, text_ds):
        '''
        load vocab, return the vectorize dataset
        set instance property:
            seld.word_to_index
        '''
        self.vectorize_layer.adapt(text_ds)
        vocab = self.vectorize_layer.get_vocabulary()  # To get words back from token indices

        dataset = text_ds.map(self.prepare_lm_inputs_labels)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.word_to_index = {}
        for index, word in enumerate(vocab):
            self.word_to_index[word] = index

        return dataset


    def prepare_lm_inputs_labels(self, text_tensor):
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
        tokenized_sentences = self.vectorize_layer(text_tensor)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y

    def make_text_gen_callback(self, start_prompt):
        start_tokens = [self.word_to_index.get(_, 1) for _ in start_prompt.split()]
        vocab = self.vectorize_layer.get_vocabulary()
        return TextGenerator(self.num_tokens_generated, self.maxlen, start_tokens, vocab)



"""
## Self-attention with causal masking
We compute self-attention as usual, but prevent any information to flow
from future tokens by masking the upper half of the scaled dot product matrix.
"""
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    @staticmethod
    def causal_attention_mask(n_dest, n_src, dtype):
        """
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        return tf.cast(m, dtype)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # prevent information flow from future tokens
        shape = tf.shape(scaled_score)
        dim_dest, dim_src = shape[2], shape[3]
        attention_mask = self.causal_attention_mask(
            dim_dest, dim_src, scaled_score.dtype
        )
        attention_mask = tf.reshape(attention_mask, [1, 1, dim_dest, dim_src])
        scaled_score = scaled_score * attention_mask - 1e4 * (1 - attention_mask)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


"""
## Implement a Transformer block as a layer
"""
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attention_output = self.att(inputs)
        attention_output = self.dropout1(attention_output)
        out1 = self.layernorm1(inputs + attention_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)


"""
## Implement embedding layer
Two seperate embedding layers, one for tokens, one for token index (positions).
"""
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

"""
## Callback for generating text
"""
class TextGenerator(keras.callbacks.Callback):
    """Callback to generate text from trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for next token
    3. Sample next token and add it to the next input
    # Arguments
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, maxlen, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.maxlen = maxlen
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def gen_text(self):
        start_tokens = [_ for _ in self.start_tokens]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.maxlen]
                sample_index = self.maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        text = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        return text

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_every != 0:
            return
        text = self.gen_text()
        print(f"generated text:\n{text}\n")
