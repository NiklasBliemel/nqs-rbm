import jax

def main():
    rng = jax.random.PRNGKey(42)
    print(rng)
    print("Hello from nqs-rbm!")


if __name__ == "__main__":
    main()
