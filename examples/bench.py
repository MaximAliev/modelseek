from core.runner import MLBench


def main():
    automl_hub = MLBench(repository='zenodo')
    dataset_repo = automl_hub.repository
    dataset_repo.load_datasets()

    automl_hub.run()


if __name__ == '__main__':
    main()