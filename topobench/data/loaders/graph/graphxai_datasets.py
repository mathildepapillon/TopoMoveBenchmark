"""Loader for the GraphXAI molecular datasets with ground-truth explanation masks."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobench.data.datasets.graphxai_dataset import GraphXAIDataset
from topobench.data.loaders.base import AbstractLoader


class GraphXAIDatasetLoader(AbstractLoader):
    """Load a GraphXAI molecular dataset (Benzene, Alkane/Fluoride-Carbonyl, Mutagenicity).

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for the processed dataset
            - data_name: One of Benzene, AlkaneCarbonyl, FluorideCarbonyl, Mutagenicity
            - raw_data_dir: Directory holding pre-staged GraphXAI ``.npz`` archives
              (optional; missing archives are downloaded from the GraphXAI repository)
            - downsample_seed: Seed for AlkaneCarbonyl class balancing (optional)
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load the dataset.

        Returns
        -------
        Dataset
            The loaded GraphXAI dataset.
        """
        return GraphXAIDataset(
            root=str(self.root_data_dir),
            name=self.parameters.data_name,
            raw_data_dir=self.parameters.get("raw_data_dir", None),
            downsample_seed=int(self.parameters.get("downsample_seed", 42)),
        )
