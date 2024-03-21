import time

COINS = [1, 5, 10, 21, 25]


def min_charge_coins_dynprog(charge: int) -> int:
    cache = {}

    for cents in range(charge + 1):
        coins_count = cents
        for coin in COINS:
            if coin <= cents:
                if cache[cents - coin] + 1 < coins_count:
                    coins_count = cache[cents - coin] + 1
        cache[cents] = coins_count

    return cache[charge]


def min_charge_coins_greedy(charge: int) -> int:
    if charge == 0:
        return 0

    result = 0
    while charge:
        for coin in reversed(COINS):
            if coin <= charge:
                result += 1
                charge -= coin
                break
    return result


charge = 5616
start = time.perf_counter()
result_dynprorg = min_charge_coins_dynprog(charge)
end = time.perf_counter()
print(f"Dynprog solution: {result_dynprorg} ({(end - start) * 1000:.2f}ms.)")

start = time.perf_counter()
result_greedy = min_charge_coins_greedy(charge)
end = time.perf_counter()
print(f"Greedy solution: {result_greedy} ({(end - start) * 1000:.2f}ms).")
