




from .hidden_variable_validator import HiddenVariableValidator
from .scatter_plot_validator import ScatterPlotValidator

validator_dict = {
    "hidden_variable_validator" : HiddenVariableValidator,
    'scatter_plot_validator' : ScatterPlotValidator
}


def get_validator(name, config):
    
    if name in validator_dict:
        return validator_dict[name](config)
    else:
        raise Exception("None validator named " + name)


