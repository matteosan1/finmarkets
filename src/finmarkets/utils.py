import pickle

from enum import IntEnum

SwapSide = IntEnum("SwapSide", {"Receiver":1, "Payer":-1, "Buyer":1, "Seller":-1})
CapFloorType = IntEnum("CapFloorType", {"Cap":1, "Floor":-1})
OptionType = IntEnum("OptionType", {"Call":1, "Put":-1})

def saveObj(filename, obj):
    """
    Utility function to pickle any "finmarkets" object

    Params:
    -------
    filename: str
        filename of the pickled object
    obj: finmarkets object
        the object to pickle
    """
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)

def loadObj(filename):
    """
    Utility function to unpickle any "finmarkets" object

    Params:
    -------
    filename: str
        filename of the object to unpickle
    """    
    with open(filename, "rb") as f:
        return pickle.load(f)

