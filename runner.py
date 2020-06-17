import argparse

from project_initiator import Initiator

def run_project():

    parser = argparse.ArgumentParser(description='DoorWay to ATiML Semester Project')
    parser.add_argument("--resource_path", help="base path of the resource folder")

    args = parser.parse_args()
    path = args.resource_path

    Initiator.start_processing(path)

if __name__ == "__main__": 
    run_project()