import numpy as np
import os
# import pyarrow.feather as feather
# import pyorc
# from pyorc.enums import StructRepr
import json

import subprocess
import sys

def install_and_import(package):
    import importlib
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except:
        print("Couldn't install")
    finally:
        globals()[package] = importlib.import_module(package)

class Decode_Preprocess:

    def __init__(self): #initialization of important data attr. 
        self.ds_names_arr=[]
        self.rootdirs=['D:/Work/PSU/Summer 2022/CSE 594/Smaller Sample'] # Change to your path
        self.avroDir = 'D:/Work/PSU/Summer 2022/CSE 594/Smaller Sample' # Change to your path

    def decode_dimacsCNF(self, ds_name): #module to parse per file, for actual decoding 
        cnf_arr=[]
        f=open(ds_name,"r")
        for line in f:
            l_check=line.split()
            allno=True
            for i in l_check:
                if not i.lstrip('-').isnumeric():
                    allno=False
                    break    
            if allno==True: #removes non pure literal statements from transferance 
                ls=line.split()
                ls1=ls[:len(ls)-1]
                if ls1!=[]:
                    cnf_subpart=np.array(list(map(int,ls1))) #each substatetement added to an instance of the problem 
                    cnf_arr.append('( ' + ' V '. join(str(x) for x in cnf_subpart) + ' )')
        f.close()

        cnf_problem=np.array(cnf_arr, dtype=object)

        label_i="" #label formation for training and testing accuracies 
        # NOTE:
        # 1 is Satisfiable
        # 0 is unsatisfiable
        # -1 is undetermined

        # Uncomment for Bigger dataset
        # if "satlib" in ds_name:
        #     if "uf20-91" in ds_name:
        #         label_i = 1
        #     elif any(word in ds_name for word in ["uf", "uuf"]):
        #         label_i = -1
        #     else:
        #         label_i = 1
        # elif "velev" in ds_name:
        #     if any(word in ds_name for word in ["unsat", "un-sat"]):
        #         label_i = 0
        #     elif "sat" in ds_name:
        #         label_i = 1
        # elif "allsat" in ds_name:
        #     label_i = 0 #UNSAT form certification for 2014 library

        # For Smaller data set
        ds_name = ds_name.lower()
        if any(word in ds_name for word in ["unsat", "uuf"]):
            label_i = 0
        elif any(word in ds_name for word in ["sat", "uf"]):
            label_i = 1
        else:
            label_i = -1 #method for satisfy_tnn labeling, unknown method yet 

        print("\nCNF DIMACS Decoding for {} instance DONE. \n".format(ds_name))

        return ' ^ '.join(cnf_problem), label_i

    def fetch_serverlib_os(self): #fetches file directories from local cloned server

        for rootDir in self.rootdirs:
            for root, dirs, files in os.walk(rootDir):
                for file in files:
                    if file.endswith(".cnf"):
                        self.ds_names_arr.append(os.path.join(root, file).replace("\\","/"))

        print("\nSERVER list of superdirectories formed. \n")

    def decode_all(self): #decodes files to NN understandable format for all files 
        filename_txt = self.avroDir + '/trial_small.avro'

        # 1. Define the schema
        schema = {
            'name': 'Data',
            'type': 'record',
            'fields': [
                {'name': 'cnf', 'type': 'string'},
                {'name': 'label', 'type': 'int'}
            ]
        }
        schema_parsed = avro.schema.parse(json.dumps(schema))
        
        with open(filename_txt, 'wb') as f:
            writer = avro.datafile.DataFileWriter(f, avro.io.DatumWriter(), schema_parsed)
            
            for i in range(len(self.ds_names_arr)):
                cnf_problem, label_i = self.decode_dimacsCNF(self.ds_names_arr[i])
                writer.append({'cnf': cnf_problem, 'label': label_i})
        
            writer.close()

        # ORC - FAILED dependency
        # with open(filename_txt, "wb") as output:
        #     with pyorc.Writer(output, "struct<cnf:string,label:int>", struct_repr=StructRepr.DICT) as writer:
        #         for i in range(len(self.ds_names_arr)):
        #             cnf_problem, label_i = self.decode_dimacsCNF(self.ds_names_arr[i])
        #             writer.write({"cnf": cnf_problem, "label": label_i})
            
        print("\nDecoding process for all CNF statements DONE. \n")  

    def complete_pre2process(self): #one fn for all operations 
        self.fetch_serverlib_os()
        self.decode_all() #all processes automated 

if __name__=="__main__":
    #Installations
    install_and_import('avro')
    install_and_import('avro.schema')
    install_and_import('avro.datafile')
    
    dp=Decode_Preprocess()
    dp.complete_pre2process() #all methods called
