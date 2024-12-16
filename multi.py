import multiprocessing
import json
import os

def dump(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def run(output_directory:str):
        p1 = multiprocessing.Process(target=dump, args=({"A":1},"A.json",output_directory, ))
        p2 = multiprocessing.Process(target=dump, args=({"B":2},"B.json",output_directory, ))
        # starting process 1
        p1.start()
        # starting process 2
        p2.start()

        # wait until process 1 is finished
        p1.join()
        # wait until process 2 is finished
        p2.join()

        # both processes finished
        print("Done!")