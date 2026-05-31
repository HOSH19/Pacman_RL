import zipfile
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filepath", help="Specify a filepath to an agent's zip file, .zip extension required.", type=str, required=True)
parser.add_argument("-s", "--savefile", help="Specify a name and location for the saved configuration file. Will be saved in .json format.", default="config")
args = parser.parse_args()

filepath = args.filepath
savefile = args.savefile

archive = zipfile.ZipFile(filepath, "r")
file = archive.open("data")
json_file = json.loads(file.read().decode("utf-8"))

# Only want to remove serialized objects from dictionary
val_to_remove = ":serialized:"

for key in json_file.keys():
    # So if each value is a type dict, then I want to iterate through it and remove the serialized key
    if type(json_file[key]) is dict:
        if val_to_remove in json_file[key].keys():
            json_file[key].pop(val_to_remove)

with open(f"{savefile}.json", "w") as outfile:
    outfile.write(json.dumps(json_file, indent=2))

file.close()
