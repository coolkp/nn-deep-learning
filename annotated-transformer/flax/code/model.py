import jax
import jax.numpy as jnp
from flax import linen as nn_flax

class InputEmbeddingsFlax(nn_flax.Module):
    d_model: int
    vocab_size: int

    @nn_flax.compact
    def __call__(self, token_matrix: jax.Array):
        embedding = nn_flax.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        return embedding(token_matrix) * jnp.sqrt(self.d_model)

