






from cvaegan import CVAEGAN


model_dict = {
    "cvaegan" : CVAEGAN,
}




def get_model(modelname, modelparams):

    if modelname in model_dict:
        return model_dict[modelname](modelparams)
    else:
        raise "No model named " + modelname



