
import random


def prepare_photon(bit, basis):
    """
    basis 0 = Rectilinear (H/V)
    basis 1 = Diagonal (D/A)
    """
    return bit, basis


def measure_photon(state, measurement_basis):
    bit, preparation_basis = state

    if preparation_basis == measurement_basis:
        return bit

    # Different bases: random measurement outcome
    return random.randint(0, 1)


def simulate_bb84(n=10000, eve=False):
    alice_bits = [random.randint(0, 1) for _ in range(n)]
    alice_bases = [random.randint(0, 1) for _ in range(n)]
    bob_bases = [random.randint(0, 1) for _ in range(n)]

    bob_results = []

    for i in range(n):

        photon = prepare_photon(
            alice_bits[i],
            alice_bases[i]
        )

        # Eve performs intercept-resend attack
        if eve:
            eve_basis = random.randint(0, 1)
            eve_bit = measure_photon(
                photon,
                eve_basis
            )

            photon = prepare_photon(
                eve_bit,
                eve_basis
            )

        bob_bit = measure_photon(
            photon,
            bob_bases[i]
        )

        bob_results.append(bob_bit)

    # Basis reconciliation
    matching = [
        i for i in range(n)
        if alice_bases[i] == bob_bases[i]
    ]

    alice_key = [alice_bits[i] for i in matching]
    bob_key = [bob_results[i] for i in matching]

    # QBER calculation
    errors = sum(
        a != b
        for a, b in zip(alice_key, bob_key)
    )

    qber = errors / len(alice_key) if alice_key else 0

    return {
        "total_photons": n,
        "sifted_key_length": len(alice_key),
        "errors": errors,
        "qber": qber
    }


if __name__ == "__main__":

    print("=== BB84 QKD Simulation ===")

    normal = simulate_bb84(eve=False)
    attacked = simulate_bb84(eve=True)

    print("\n[Normal Quantum Channel]")
    for key, value in normal.items():
        print(f"{key}: {value}")

    print("\n[Intercept-Resend Attack]")
    for key, value in attacked.items():
        print(f"{key}: {value}")
