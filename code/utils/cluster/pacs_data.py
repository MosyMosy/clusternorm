from typing import Optional
import os
from typing import Optional, Callable, Tuple, Any, List
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive

__all__ = ["pacs_art","pacs_cartoon","pacs_photo","pacs_sketch"]

def download_data(root: str, file_name: str, archive_name: str, url_link: str):
    """
    Download file from internet url link.

    Args:
        root (str) The directory to put downloaded files.
        file_name: (str) The name of the unzipped file.
        archive_name: (str) The name of archive(zipped file) downloaded.
        url_link: (str) The url link to download data.

    .. note::
        If `file_name` already exists under path `root`, then it is not downloaded again.
        Else `archive_name` will be downloaded from `url_link` and extracted to `file_name`.
    """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Downloading {}".format(file_name))
        # if os.path.exists(os.path.join(root, archive_name)):
        #     os.remove(os.path.join(root, archive_name))
        try:
            download_and_extract_archive(url_link, download_root=root, filename=archive_name, remove_finished=False)
        except Exception:
            print("Fail to download {} from url link {}".format(archive_name, url_link))
            print('Please check you internet connection.'
                  "Simply trying again may be fine.")
            exit(0)

def check_exits(root: str, file_name: str):
    """Check whether `file_name` exists under directory `root`. """
    if not os.path.exists(os.path.join(root, file_name)):
        print("Dataset directory {} not found under {}".format(file_name, root))
        exit(-1)

def read_list_from_file(file_name: str) -> List[str]:
    """Read data from file and convert each line into an element in the list"""
    result = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            result.append(line.strip())
    return result

class ImageList(datasets.VisionDataset):
    """A generic Dataset class for image classification

    Args:
        root (str): Root directory of dataset
        classes (list[str]): The names of all the classes
        data_list_file (str): File to read the image list from.
        transform (callable, optional): A function/transform that  takes in an PIL image \
            and returns a transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `data_list_file`, each line has 2 values in the following format.
        ::
            source_dir/dog_xxx.png 0
            source_dir/cat_123.png 1
            target_dir/dog_xxy.png 0
            target_dir/cat_nsdf3.png 1

        The first value is the relative path of an image, and the second value is the label of the corresponding image.
        If your data_list_file has different formats, please over-ride :meth:`~ImageList.parse_data_file`.
    """

    def __init__(self, root: str, classes: List[str], data_list_file: str,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self.parse_data_file(data_list_file)
        self.classes = classes
        self.class_to_idx = {cls: idx
                             for idx, cls in enumerate(self.classes)}
        self.loader = default_loader
        self.data_list_file = data_list_file

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Args:
            index (int): Index
            return (tuple): (image, target) where target is index of the target class.
        """
        path, target = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and target is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self) -> int:
        return len(self.samples)

    def parse_data_file(self, file_name: str) -> List[Tuple[str, int]]:
        """Parse file to data list

        Args:
            file_name (str): The path of data file
            return (list): List of (image path, class_index) tuples
        """
        with open(file_name, "r") as f:
            data_list = []
            for line in f.readlines():
                split_line = line.split()
                target = split_line[-1]
                path = ' '.join(split_line[:-1])
                if not os.path.isabs(path):
                    path = os.path.join(self.root, path)
                target = int(target)
                data_list.append((path, target))
        return data_list

    @property
    def num_classes(self) -> int:
        """Number of classes"""
        return len(self.classes)

    @classmethod
    def domains(cls):
        """All possible domain in this dataset"""
        raise NotImplemented

class PACS(ImageList):
    """`PACS Dataset <https://domaingeneralization.github.io/#data>`_.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            art_painting/
                dog/
                    *.jpg
                    ...
            cartoon/
            photo/
            sketch
            image_list/
                art_painting.txt
                cartoon.txt
                photo.txt
                sketch.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/add42cc3859847bc988c/?dl=1"),
        ("art_painting", "art_painting.tgz", "https://cloud.tsinghua.edu.cn/f/4eb7db4f3eec41719856/?dl=1"),
        ("cartoon", "cartoon.tgz", "https://cloud.tsinghua.edu.cn/f/d847ac22497b4826889f/?dl=1"),
        ("photo", "photo.tgz", "https://cloud.tsinghua.edu.cn/f/458ad21483da4a45935b/?dl=1"),
        ("sketch", "sketch.tgz", "https://cloud.tsinghua.edu.cn/f/c892ac2d94a44b1196b8/?dl=1"),
    ]
    image_list = {
        "A": "image_list/art_painting_{}.txt",
        "C": "image_list/cartoon_{}.txt",
        "P": "image_list/photo_{}.txt",
        "S": "image_list/sketch_{}.txt"
    }
    CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    def __init__(self, root: str, task: str, train: bool = True, transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None, download: bool = False):
       
        assert task in self.image_list
        
        if train:
            split = "train"
        else:
            split = "val"
        assert split in ["train", "val", "all", "test"]
            
        data_list_file = os.path.join(root, self.image_list[task].format(split))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(PACS, self).__init__(root, PACS.CLASSES, data_list_file=data_list_file, transform= transform,
            target_transform= target_transform)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())

class pacs_art(PACS):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, "A", train, transform, target_transform, download)

class pacs_cartoon(PACS):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, "C", train, transform, target_transform, download)

class pacs_photo(PACS):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, "P", train, transform, target_transform, download)
        
class pacs_sketch(PACS):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False):
        super().__init__(root, "S", train, transform, target_transform, download)