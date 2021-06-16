def get_gm(acc, cs, ppl):
    return (max(acc, 0) * max(cs, 0) * max(1 / ppl, 0)) ** (1 / 3)