import math
from llm import pass_to_llm
from config import standard_size,input_text
from npl_utils import measure_length,split_into_slices

def main(input_text):

    input_size = measure_length(input_text)
    
    if input_size <= standard_size:
        # Pass the input "as it is" to the LLM
        final_content=pass_to_llm(input)
        return final_content
    else:
        # Split the input into slices and process each slice
        result=[]
        slices = split_into_slices(input_text)
        for item in slices:
            print (item)
        for slice in slices:
            result.append(pass_to_llm(slice))
    return result

def test_slicing():
    input_size = measure_length(input_text)
    print(input_size)
    if input_size <= standard_size:
        print(input_text)
    else:
        result=[]
        slices = split_into_slices(input_text)
        for item in slices:
            print (item)
        for slice in slices:
            result.append(pass_to_llm(slice))
        slices = split_into_slices(input_text)
        for item in slices:
            print(item)
test_slicing()
# main(input_text)
