import glob
import csv
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

# Load metadata etc
filenames_and_labels = r'F:\temp\thesisdata\SAATCHI_DATASET_FULL.tsv'
delimiter = '\t'
image_input_folder = r'C:\Users\R\PycharmProjects\Thesis_Saatchi_scraper\micro_dataset1'
image_output_folder = r'F:\temp\thesisdata\saatchi_micro_resized512'
size_ = 512


def resize_and_pad_image(input_path: str,
                         output_path: str,
                         desired_size: int):
    input_path_ = Path(input_path)
    output_path_ = Path(output_path)

    assert input_path_.is_file()
    assert output_path_.is_dir(), print('Supplied output path is not a directory:' + output_path_.__str__())

    if input_path_.stat().st_size > 0:
        pass
    else:
        print(f'Filesize is 0, skipping file: {input_path_}')
        return

    filename = input_path_.name
    img = Image.open(input_path)
    old_size = img.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_img = Image.new('RGB', (desired_size, desired_size))
    new_img.paste(img, ((desired_size - new_size[0]) // 2,
                        (desired_size - new_size[1]) // 2))

    full_output_path = output_path_ / filename
    new_img.save(full_output_path)


# Create a dict with all filenames and associated labels
with open(filenames_and_labels, 'rt')as f:
    data = list(csv.reader(f, delimiter=delimiter))
    file_dict = {}
    for row in data:
        file_dict.update({row[0]: row[1]})

# Create list of sanitized labels to be used as folder names
label_folder_list = [s.replace(' ', '_')
                     .replace('&', '_')
                     .replace('/', '_')
                     .replace('__', '_')
                     .replace('__', '_')
                     .lower()
                     for s in set(file_dict.values())]

# Create dict for lookup up the correct folder name given a label
label_folder_lookup = {}

for s in set(file_dict.values()):
    label_folder_lookup.update({s: s.replace(' ', '_')
                                    .replace('&', '_')
                                    .replace('/', '_')
                                    .replace('__', '_')
                                    .replace('__', '_')
                                    .lower()})
print(f'Lookup dict: {label_folder_lookup}')

# Create the folders
for folder in label_folder_list:
    Path(image_output_folder + '/' + folder).mkdir(parents=True, exist_ok=True)

print('Resizing and moving files...')
for file in tqdm(glob.glob(image_input_folder + '*/*')):
    try:
        image_output_folder_with_label = image_output_folder + '\\' + label_folder_lookup[file_dict[Path(file).name]]
        resize_and_pad_image(file, image_output_folder_with_label, size_)
    except KeyError:
        print(f'Label not found for file {file}, skipping!')
