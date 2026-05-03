from src.modelfast.domain import Dataset
from src.modelfast.api import Modeler
from src.modelfast.repository import ImbalancedDatasetRepository

def main():
    datasets = ImbalancedDatasetRepository(verbosity=2).load_datasets()

    modelfind = Modeler(
        backend='modelfast',
        metric='f1',
        extra_metrics=['precision', 'recall'],
        timeout=3600,
        verbosity=1
    )

    for dataset in datasets:
        modelfind.run(dataset)


if __name__ == '__main__':
    main()