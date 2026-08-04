from dual_uq.manifests import initialize_manifests


def main() -> None:
    initialize_manifests("data/manifests")
    print("Initialized manifest tables in data/manifests/")
if __name__ == "__main__":
    main()
