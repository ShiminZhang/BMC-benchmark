import os
from .paths import get_aig_dir

def get_all_instance_names():
    aig_dir = get_aig_dir()
    for file in os.listdir(aig_dir):
        if file.endswith(".aig"):
            yield file.split(".")[0]

def main():
    test_names = get_all_instance_names()
    for name in test_names:
        print(name)

if __name__ == "__main__":
    main()


