import streamlit as st
from openai import OpenAI
import os
import tempfile
import io
from pyomo.environ import *
import enum
from Util import load_model, extract_component, add_eg, read_iis
from Util import infer_infeasibility, param_in_const, extract_summary, evaluate_gpt_response, classify_question, get_completion_detailed, convert_to_standard_form, get_constraints_n_parameters, get_completion_for_quantity_sensitivity, get_variables_n_indices, get_completion_general_stream, string_generator, extract_comments
from Util import get_completion_from_messages_withfn, gpt_function_call, get_parameters_n_indices, get_completion_for_index, get_completion_for_index_sensitivity, get_constraints_n_indices, get_completion_general, get_completion_from_messages_withfn_its, get_completion_for_index_variables, store_and_load_index, run_my_code
from get_code_from_markdown import *
from llama_index.llms.openai import OpenAI as oai
from llama_index.core import Settings
import contextlib
import time
import sys




import importlib

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.set_page_config(layout='wide')


def callback(prompt):
    initial_code = st.session_state['pyomo_model_engine'].query(prompt)
    comments, code_blocks = extract_comments(initial_code.response, st.session_state.model)
    print("code blocks", code_blocks)
    if len(code_blocks) == 0:
        new_code = st.session_state['pyomo_source_engine'].query(prompt)
    
    new_code = st.session_state['pyomo_source_engine'].query(code_blocks[0])
    print(' '.join(comments).replace('#', ''))
    print(new_code.response, file=open('new_code.txt', 'a'))
    model = st.session_state['model']
    # logs = run_my_code(initial_code.response, st.session_state.model)
    # print('logs', logs, file=open('logs.txt', 'a'))
    if "# The code is correct" in new_code.response:
        model = st.session_state['model']
        blocks = get_code_from_markdown([initial_code.response])
        print("blocks", blocks)
        f = StringIO()
        # with stdoutIO() as s:
        with redirect_stdout(open('out','w')):
            for block in blocks:
                exec(block)
                sys.stdout.flush()
                time.sleep(1)
            # run_code_from_markdown_blocks(blocks, method=RunMethods.EXECUTE)
        logs = open('out').read()
        print(logs, file=open('logs.txt', 'a'))
    else:
        model = st.session_state['model']
        blocks = get_code_from_markdown([new_code.response])
        print("new blocks", blocks)
        f = StringIO()
        with redirect_stdout(open('out','w')):
            for block in blocks:
                exec(block)
                sys.stdout.flush()
                time.sleep(1)
    
        logs = open('out').read()
        comments, code_blocks = extract_comments(new_code.response, st.session_state.model)
        print('new logs', logs, file=open('new_logs.txt', 'a'))
    return logs, comments


class Question_Type(enum.Enum):
    ITS = "1"
    SEN = "2"
    GEN = "3"
    DET = "4"
    OPT = "5"
    OTH = "6"

st.sidebar.title("Settings")
# st.sidebar.subheader("Choose your model")
gpt_model = st.sidebar.selectbox(label="GPT-Model", options=["gpt-4-turbo-preview", "gpt-4-1106-preview", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"], )

st.session_state["gpt_model"] = gpt_model
llm = oai(model=st.session_state["gpt_model"], api_key=os.environ["OPENAI_API_KEY"])
Settings.llm = llm
st.sidebar.subheader("Load Pyomo File")
uploaded_file = st.sidebar.file_uploader("Upload", type=["py"])



def process_file():
    if uploaded_file:
        py_path = os.path.abspath(uploaded_file.name)
        if not os.path.isdir("../saved_files"):
            os.mkdir("../saved_files")
        with tempfile.TemporaryDirectory() as tmpdirname:
        # Construct the file path within the temporary directory
            file_path = os.path.join("../saved_files/", uploaded_file.name)
            
            # Write the uploaded file's contents to the temporary file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # py_path = uploaded_file.name
            print(file_path)
        model, ilp_path = load_model(file_path)
        st.session_state['model'] = model
        const_list, param_list, var_list, PYOMO_CODE = extract_component(st.session_state['model'], file_path)
        st.session_state['param_names'] =  param_list
        print(f"type of param_names: {type(st.session_state['param_names'])}")
        print(f"names of param_names: {st.session_state['param_names']}")
        summary = extract_summary(var_list, param_list, const_list, PYOMO_CODE, st.session_state['gpt_model'])
        
        st.session_state['const_list'] = const_list
        st.session_state['param_list'] = param_list
        st.session_state['PYOMO_CODE'] = PYOMO_CODE
        st.session_state['ilp_path'] = ilp_path
        st.session_state['py_path'] = file_path
        st.session_state['table'] = summary


        st.session_state['model_info'] = get_parameters_n_indices(model)
        st.session_state['model_constraint_info'] = get_constraints_n_indices(model)
        st.session_state['model_constraint_parameters_info'] = get_constraints_n_parameters(model)
        st.session_state['model_variable_info'] = get_variables_n_indices(model)
        
        st.session_state.messages.append({
            "role": "system",
            "content": summary
        })

        summary_response = add_eg(summary, st.session_state['gpt_model'])
        st.session_state['summary'] = summary_response

        const_names, param_names, iis_dict = read_iis(st.session_state['ilp_path'], st.session_state.model)
        iis_relation = param_in_const(iis_dict)
        st.session_state['relation'] = iis_relation
        infeasibility_report = infer_infeasibility(const_names, param_names, summary, st.session_state['gpt_model'], st.session_state['model'])
        st.session_state['infeasibility_report'] = infeasibility_report
        
        st.session_state.messages.append({
            'role': 'assistant',
            'content': st.session_state['summary'] + '\n\n' + infeasibility_report
        })

        pyomo_model_engine, pyomo_source_engine, namespace = store_and_load_index(st.session_state['py_path'], st.session_state['summary'] + '\n\n' + infeasibility_report)
        st.session_state['pyomo_model_engine'] = pyomo_model_engine
        st.session_state['pyomo_source_engine'] = pyomo_source_engine
        st.session_state['namespace'] = namespace


        log_files = ['./initial_code.txt', './logs.txt', './new_code.txt', './new_logs.txt']

        for log_file in log_files:
            with open(log_file, "w") as f:
                f.truncate()
        

    else:
        st.write("Please upload a file")
        return




st.sidebar.button("Process", on_click=process_file)


st.title("Infeasible Model Troubleshooter")

# Set a default model
if "gpt_model" not in st.session_state:
    st.session_state["gpt_model"] = "gpt-4-turbo-preview"

# Initialize the Chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {'role': 'system',
                                  'content': """You are an expert in optimization and Pyomo who helps unskilled user to 
                                  troubleshoot the infeasible optimization model and any optimization questions. \n
                                  You are encouraged to remind users that they can change the value of model parameters to 
                                  make the model become feasible, but try your best to avoid those parameters that have 
                                  product with variables. \nIf the users ask you to change a parameter that has product 
                                  with variable, DO NOT use "they are parameters that have product with variables" as 
                                  explanation. Instead, you should give the physical or business context to explain why 
                                  this parameter cannot be changed. If the users keep insisting on changing the 
                                  parameter, you can try changing them but give them a warning. \n
                                  You are not allowed to have irrelevant conversation with users."""}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message['role'] in  ['user', 'assistant']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

from llama_index.core import PromptTemplate
from io import StringIO
from contextlib import redirect_stdout

# Accept user input
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with open(st.session_state.py_path, 'r') as f:
        file_contents = f.read()


    classification = classify_question(st.session_state.messages[-1], st.session_state['gpt_model'])
    print(f"classification: {classification}")
    # # print([_ for _ in st.session_state.model.component_objects(Param)])
    # model_ = st.session_state.model
    # if classification == Question_Type.ITS.value:
    #     response = get_completion_from_messages_withfn_its(st.session_state.messages, st.session_state['gpt_model'])
    #     try:
    #         print("hi")
    #         print(response)
    #         fn_call = response.choices[0].message.tool_calls[0]
    #         fn_name = fn_call.function.name
    #         arguments = fn_call.function.arguments
    #         args_dict = eval(arguments)
    #         print(args_dict.keys())
    #         param_names = args_dict.get("param_names", None)
    #         print(param_names)
    #         print(eval("st.session_state.model." + param_names[0] + ".is_indexed()"))
    #         for param_name in param_names:
    #             if eval(f"st.session_state.model.{param_name}.is_indexed()"):
    #                 print(f"param_name: {param_name}")
    #                 new_response = get_completion_for_index(st.session_state.messages[-1], st.session_state.model_info, st.session_state.PYOMO_CODE, st.session_state.gpt_model)
    #                 break
    #             else:
    #                 new_response = response
    #         print(f"new_response: {new_response}")
    #         (fn_message, flag), fn_name = gpt_function_call(new_response, st.session_state.param_names, st.session_state.model)
    #         orig_message = {'role': 'function', 'name': fn_name, 'content': fn_message}
    #         st.session_state.messages.append(orig_message)
    #         print(f"flag: {flag}")
    #         if flag == 'feasible':
    #             expl_message = {'role': 'system',
    #                             'content': 'Tell the user that you made some changed to the code and ran it, and '
    #                                     'the model becomes feasible. '
    #                                     'Replace the parameter symbol in the text with its physical meaning and mention the amount by which you changed it (if applicable)'
    #                                     '(for example, you could say "increasing the amount of cost invested" '
    #                                     'instead of saying "increasing c") '
    #                                     'and provide brief explanation.'}
    #         elif flag == 'infeasible':
    #             expl_message = {'role': 'system',
    #                             'content': 'Tell the user that you made some changed to the code and ran it, but '
    #                                     'the model is still infeasible. '
    #                                     'Explain why it does not become feasible and '
    #                                     'suggest other parameters that the user can try.'}
    #         elif flag == 'invalid':
    #             expl_message = {'role': 'system',
    #                             'content': 'Tell the user that you cannot change the things they requested. '
    #                                     'Explain why users instruction is invalid and '
    #                                     'suggest the parameters that the user can try.'}
    #         st.session_state.messages.append(expl_message)
    #         stream = get_completion_general_stream(st.session_state.messages, st.session_state.gpt_model)
    #         with st.chat_message("assistant"):
    #             response = st.write_stream(stream)
    #             st.session_state.messages.append({"role": "assistant", "content": response})
    #     except:
    #         new_response = response.choices[0].message.content
            
    #         st.session_state.messages.append({'role': 'assistant', 'content': new_response})
    #         stream = string_generator(new_response)
    #         with st.chat_message("assistant"):
    #             st.write_stream(stream)
    
    
    # elif classification == Question_Type.SEN.value:
    #     new_response = get_completion_for_index_sensitivity(st.session_state.messages[-1], st.session_state.model_info, st.session_state.model_constraint_parameters_info, st.session_state.PYOMO_CODE, st.session_state.gpt_model)
    #     (fn_message, flag), fn_name = gpt_function_call(new_response, st.session_state.param_names, st.session_state.model, nature='sensitivity_analysis', user_query=st.session_state.messages[-1], gpt_model=st.session_state.gpt_model)
    #     orig_message = {'role': 'function', 'name': fn_name, 'content': fn_message}
    #     st.session_state.messages.append(orig_message)
    #     if flag == 'feasible':
    #         #  TODO: Done
    #         expl_message = {'role': 'system',
    #                         'content': 'Tell the user that you did sensitivity analysis for the parameter they asked'
    #                                     'Replace the parameter symbol in the text with its physical meaning '
    #                                     '(for example, you could say "changing the number of men in each port" '
    #                                     'instead of saying "increasing demand_rule") '
    #                                     'and provide brief explanation.'}
    #     elif flag == 'infeasible':
    #         expl_message = {'role': 'system',
    #                         'content': 'Tell the user that you resolved the model after making the requested changes.'
    #                                     'But it turned out that the model is now infeasible.'
    #                                     'Explain why it is not possible to perform sensitivity analysis for an infeasible model and '
    #                                     'suggest other ways that the user can try.'}
    #     elif flag == 'invalid':
    #         expl_message = {'role': 'system',
    #                         'content': 'Tell the user that you cannot change the things they requested. '
    #                                     'Explain why users instruction is invalid and '
    #                                     'suggest the parameters that the user can try.'}
    #     st.session_state.messages.append(expl_message)
    #     stream = get_completion_general_stream(st.session_state.messages, st.session_state.gpt_model)
    #     with st.chat_message("assistant"):
    #         response = st.write_stream(stream)
    #         st.session_state.messages.append({"role": "assistant", "content": response})

    # elif classification == Question_Type.GEN.value:
    #     stream = get_completion_general_stream(st.session_state.messages, st.session_state['gpt_model'])
    #     with st.chat_message("assistant"):
    #         response = st.write_stream(stream)
    #         st.session_state.messages.append({"role": "assistant", "content": response})
    
    # elif classification == Question_Type.DET.value:
    #     stream = get_completion_detailed(st.session_state.messages[-1], st.session_state.model_info, st.session_state.PYOMO_CODE, st.session_state.gpt_model)
    #     with st.chat_message("assistant"):
    #         response = st.write_stream(stream)
    #         st.session_state.messages.append({"role": "assistant", "content": response})
    
    # elif classification == Question_Type.OPT.value:
    #     new_response = get_completion_for_index_variables(st.session_state.chatbot_messages[-1], st.session_state.model_variables_info, st.session_state.PYOMO_CODE, st.session_state.gpt_model)
    #     (fn_message, flag), fn_name = gpt_function_call(new_response, st.session_state.param_names, st.session_state.model, nature='optimal_value', user_query=st.session_state.chatbot_messages[-1], gpt_model=st.session_state.gpt_model)
    #     orig_message = {'role': 'function', 'name': fn_name, 'content': fn_message}
    #     st.session_state.messages.append(orig_message)
    #     if flag == 'feasible':
    #         expl_message = {'role': 'system',
    #                         'content': 'Tell the user that you ran the model and found the optimal value for the variables and the objective function they asked'
    #                                     'Replace the objective name and variable names in the text with its physical meaning '
    #                                     '(for example, you could say "the minimum budget" '
    #                                     'instead of saying "obj") '
    #                                     'and provide brief explanation.'}
    #     elif flag == 'infeasible':
    #         expl_message = {'role': 'system',
    #                         'content': 'Tell the user that the model is infeasible and hence you cannot find the optimal value for the variables and the objective function they asked'
    #                                     'Explain why it is not possible to find the optimal value for the variables and the objective function for an infeasible model and '
    #                                     'suggest other ways that the user can try.'}
    #     elif flag == 'invalid':
    #         expl_message = {'role': 'system',
    #                         'content': 'Tell the user that you cannot answer what they requested'
    #                                     'Explain why users instruction is invalid and '
    #                                     'suggest they can ask instead.'}
    #     st.session_state.messages.append(expl_message)
    #     stream = get_completion_general_stream(st.session_state.messages, st.session_state.gpt_model)
    #     with st.chat_input("assistant"):
    #         response = st.write_stream(stream)
    #         st.session_state.messages.append({"role": "assistant", "content": response})

    # elif classification == Question_Type.OTH.value:
        # expl_message = {
        #         'role': 'system',
        #         'content': """Tell the user that you do not have the capability to answer this kind of queries yet.
        #         Explain it to the user that you can help with any other queries regarding the model information general,
        #         infeasibility troubleshooting and sensitivity analysis"""

        #     }
        # st.session_state.messages.append(expl_message)
        # stream = get_completion_general_stream(st.session_state.messages, st.session_state.gpt_model)
        # with st.chat_message("assistant"):
        #     response = st.write_stream(stream)
        #     st.session_state.messages.append({"role": "assistant", "content": response})
   
    code = get_code_from_markdown(file_contents)
    for co in code:
        exec(co, globals())
    print("prompt: ", prompt)

    initial_code = st.session_state['pyomo_model_engine'].query(prompt)
    
    print(initial_code.response, file=open('initial_code.txt', 'a'))

    comments, code_blocks = extract_comments(initial_code.response, st.session_state.model)
    print("code blocks", code_blocks)
    if len(code_blocks) == 0:
        new_code = st.session_state['pyomo_source_engine'].query(prompt)
    new_code = st.session_state['pyomo_source_engine'].query(code_blocks[0])
    print(' '.join(comments).replace('#', ''))
    print(new_code.response, file=open('new_code.txt', 'a'))

    model = st.session_state['model']
    # logs = run_my_code(initial_code.response, st.session_state.model)
    # print('logs', logs, file=open('logs.txt', 'a'))
    if "# The code is correct" in new_code.response:
        model = st.session_state['model']
        blocks = get_code_from_markdown([initial_code.response])
        print("blocks", blocks)
        f = StringIO()
        FLAG = False
        # with stdoutIO() as s:
        with redirect_stdout(open('out','w')):
            for block in blocks:
                try:
                    exec(block)
                    sys.stdout.flush()
                    time.sleep(1)
                except:
                    FLAG = True
                    break
        if FLAG:
            try:
                logs, comments = callback(prompt)
            except:
                expl_message = {'role': 'system',
                    'content': f""" Tell the user to re-frame the question and ask it again because it was not clear. 
                    """                    
                    }

                st.session_state.messages.append(expl_message)

                with st.chat_message("assistant"):
                    stream = client.chat.completions.create(
                        model=st.session_state["gpt_model"],
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                    )
                    response = st.write_stream(stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            # run_code_from_markdown_blocks(blocks, method=RunMethods.EXECUTE)
        else:
            logs = open('out').read()
        print(logs, file=open('logs.txt', 'a'))
    else:
        model = st.session_state['model']
        blocks = get_code_from_markdown([new_code.response])
        print("new blocks", blocks)
        f = StringIO()
        FLAG = False
        # with stdoutIO() as s:
        with redirect_stdout(open('out','w')):
            for block in blocks:
                try:
                    exec(block)
                    sys.stdout.flush()
                    time.sleep(1)
                except:
                    FLAG = True
                    break
        if FLAG:
            logs, comments = callback(prompt)

        else:

            # run_code_from_markdown_blocks(blocks, method=RunMethods.EXECUTE)
            logs = open('out').read()
        comments, code_blocks = extract_comments(new_code.response, st.session_state.model)
        print('new logs', logs, file=open('new_logs.txt', 'a'))
    
    expl_message = {'role': 'system',
                    'content': f""" The engineer has written a small python script to answer the user's query.
                    Here are the comments from the code: {' '.join(comments).replace('#', '')}.
                    Here are the logs from the code: {logs}.

                    Using the logs and the comments, you have to frame a nice response to the user. 
                    DO NOT MENTION ANYTHING ABOUT THE ENGINEER AND THE CODE.
                    """                    
                    }

    st.session_state.messages.append(expl_message)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["gpt_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
