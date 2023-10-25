# Gurobi
import typing
import os
import sys
import importlib
import pyomo.environ as pe
from pyomo.opt import SolverFactory
from pyomo.contrib.iis import *
from pyomo.core.expr.visitor import identify_mutable_parameters, replace_expressions, clone_expression
# GPT
import openai
import tiktoken
import json
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']



def get_completion_standalone(prompt, gpt_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


def load_model(pyomo_file):
    original_dir = os.getcwd()
    directory_path = os.path.dirname(pyomo_file)
    filename_wo_extension = os.path.splitext(os.path.basename(pyomo_file))[0]
    sys.path.append(directory_path)

    module = importlib.import_module(filename_wo_extension)
    model = module.model  # access the pyomo model (remember to name your model as 'model' eg. model = RTN)
    print(f'Model {pyomo_file} loaded')
    try:
        ilp_name = write_iis(model, filename_wo_extension + ".ilp", solver="gurobi")
        ilp_path = os.path.abspath(filename_wo_extension + ".ilp")
    except Exception as e:
        print(e.message)
        if e.message ==  "Cannot compute IIS on a feasible model":
            ilp_path = ""
    return model, ilp_path


def extract_component(model, pyomo_file):
    const_list = []
    param_list = []
    var_list = []
    idx_param_list = []
    for const in model.component_objects(pe.Constraint):
        const_list.append(str(const))
    for param in model.component_objects(pe.Param):
        if param.is_indexed():
            param_list.append(str(param))
        else:
            param_list.append(str(param))
    for var in model.component_objects(pe.Var):
        var_list.append(str(var))
    
    with open(pyomo_file, 'r') as file:
        PYOMO_CODE = file.read()
    file.close()
    return const_list, param_list, idx_param_list, var_list, PYOMO_CODE


def extract_summary(var_list, param_list, idx_param_list, const_list, PYOMO_CODE, gpt_model):
    prompt = f"""Here is an optimization model written in Pyomo, which is delimited by triple backticks. 
    Your task is to 
    (1): use plain English to describe the objective funtion of this model. \n\n
    (2): We identified that it includes variables: {var_list}, please output a table and each row is in a style of 
    - <Name of the variable> | <physical meaning of the variable>. \n\n
    (3) We identified that it includes parameters: {param_list}, please output a table and each row is in a style of
    - <Name of the parameter> | <physical meaning of the parameter>. \n\n
    (4) We identified that it includes constraints: {const_list} please output a table and each row is in a style of 
    - <Name of the constraint> | <physical meaning of the constraint>. 
    You need to cover the physical meaning of each term in the constraint expression and give a detailed explanation. \n\n
    (5) Identify the parameters that have product with variables in constraints. 
    For example, suppose "a" is a parameter and "b" is a variable, if a*b is in the constraint, then a is the parameter that 
    has product with variables in constraints.
    
    Pyomo Model Code: ```{PYOMO_CODE}```"""
    summary_response = get_completion_standalone(prompt, gpt_model)
    return summary_response


def add_eg(summary, gpt_model):
    prompt = f"""I will give you a decription of an optimization model with parameters, variables, constraints and objective. 
    First introduce this model to the user using the following four steps. However, DO NOT write bullets 1-4\
        make it more sounds like coherent paragraph:
                                      1. Try to guess what the problem is about and who is using is model for deciding 
                                      what problem.\
                                        give a high level summary, e.g. "An oil\
                                        producer has developed an optimization to determine where to drill the wells".
                                        "A travel planner is determining the best way to visit n cities".explain what data is available to the decision maker\
                                            make decisions in plain English. Avoid using bullet points!
                                      Try to make it smooth like a story. 
                                        for example you could say "You are given a number of cities and the distance between any two
                                            cities." for a TSP problem. You can say "You are given n item with different values and
                                                weights to be filled in a knapsack who capacity is known"
                                      2. explain what decisions are to be made in plain English. Avoid using bullet points!
                                      Try to make it smooth like a story. \
                                        for example, you could say "you would like to decide the sequence to visit all the n cities." for the TSP 
                                        problem.
                                        you could say "you would like to decide the items to be filled in the knapsack" for the knapsack problem. 
                                    3. explain what constraints the decisions have to satisfy in plain English
                                        for example you could say "the weights of all the items in the knapsack have to be less than or 
                                        equal to the knapsack capacity"
                                    4. explain the objective function in plain English
                                        you could say "given these decisions, we would like to find the shortest path" for the TSP problem.
                                        "given these decisions and constraints, we would like to find the items to be filled in the knapsack that 
                                        have the total largest values"               
    Model Description: ```{summary}```"""
    summary_response = get_completion_standalone(prompt, gpt_model)
    return summary_response


def read_iis(ilp_file, model):
    constr_names = []
    iis_dict = {}
    param_names = []
    try:
        with open(ilp_file, 'r') as file:
            ilp_string = file.read()
        file.close()
        ilp_lines = ilp_string.split("\n")
        for iis_line in ilp_lines:
            if ":" in iis_line:
                constr_name = iis_line.split(":")[0].split("(")[0]
                if constr_name not in constr_names:
                    constr_names.append(constr_name)

        for const_name in constr_names:
            iis_dict.update({const_name: []})
            consts = eval('model.' + const_name)
            for const_idx in consts:
                const = consts[const_idx]
                expr_parameters = identify_mutable_parameters(const.expr)
                for p in expr_parameters:
                    p_name = p.name.split("[")[0]
                    param_names.append(p_name)

                    if p_name not in iis_dict[const_name]:
                        iis_dict[const_name].append(p_name)

        param_names = list(set(param_names))
    except Exception as e:
        # Model is feasible
        print(e)
        for constr in model.component_objects(pe.Constraint):
            constr_names.append(constr._name)
        for const_name in constr_names:
            iis_dict.update({const_name: []})
            consts = eval('model.' + const_name)
            for const_idx in consts:
                const = consts[const_idx]
                expr_parameters = identify_mutable_parameters(const.expr)
                for p in expr_parameters:
                    p_name = p.name.split("[")[0]
                    param_names.append(p_name)

                    if p_name not in iis_dict[const_name]:
                        iis_dict[const_name].append(p_name)

        param_names = list(set(param_names))
    return constr_names, param_names, iis_dict


def param_in_const(iis_dict):
    text_list = []
    for key, values in iis_dict.items():
        if values:
            if len(values) == 1:
                text_list.append(f"{key} constraint only contains {values[0]} parameter")
            else:
                objects = ', '.join(values[:-1]) + f" and {values[-1]}"
                text_list.append(f"{key} constraint contains {objects} parameters")
        else:
            text_list.append(f"{key} constraint contains no parameter")

    final_text = ', '.join(text_list) + '.\n'
    return final_text

def infer_feasibility(const_names, param_names, summary, gpt_model):
    prompt = f"""Optimization experts are troubleshooting an optimization model. They found that the model is 
    feasible. They found that {', '.join(const_names)} constraints are present in the model and that
    {', '.join(param_names)} are the parameters involved in these constraints. To understand what the parameters
    and the constraints mean, here's the model summary in a Markdown Table ```{summary}```\
    Now, given these information, your job is to do the following steps. Try to use plain english!
    DO NOT show "A-B", show the answers in two paragraphs:
    A. Tell the user something like "The following constraints are present in the model. Then provide the list
    of constraints ({', '.join(const_names)}) and their physical meaning in an itemized list. You can refer to the
    model summary I gave you to get the meaning of the constraints. Avoid using any symbols of the constraints, use
    natural language. For example, answer to this step can be
    "The following constraints are present in the model:
    C1. The mass balance constraints that specify the level of the storage vessel at a give time point\
        is equal to the
    C2. The storage level should be less than its maximum capacity.
    "
    B. Tell the user all the parameters, {', '.join(param_names)} \
        involved in the constraints and their physical meaning in an itemized list.
        You can refer to the model summary I gave you to get the meaning of the parameters. \
            Avoid using any symbols of the parameters. For example, answer to this step can be
            "The following input data are involved in the constraints:
            P1. The molecular weight of a molecule A
            P2. the demand of customers
            P3. the storage capacity"
    """
    explanation = get_completion_standalone(prompt, gpt_model)
    return explanation

def infer_infeasibility(const_names, param_names, summary, gpt_model, model):
    prompt = f"""Optimization experts are troubleshooting an infeasible optimization model. 
    They found that {', '.join(const_names)} constraints are in the Irreducible infeasible set.
    and that  {', '.join(param_names)} are the parameters involved in the Irreducible infeasible set.
    To understand what the parameters and the constraints mean, Here's the  Model Summary \
        in a Markdown Table ```{summary}```\
    Now, given these information, your job is to do the following steps. Try to use plain
    english! DO NOT show "A-C", show the answers in three papagraphs:
    A. Tell the user something like "The following constraints are causing the model to be infeasible". 
    Then provide the list constraints ( {', '.join(const_names)}) and their physical meaning in an itemized list.
    You can refer to the Model Summary I gave you to get the meaning of the constraints. Avoid using any
    symbols of the constraints, use natural language. For example, answer to this step can be 
    "The following constraints are causing the model to be infeasible:
    C1. The mass balance constraints that specify the level of the storage vessel at a given time point\
        is equal to the 
    C2. The storage level should be less than its maximum capacity.
    "
    B. Tell the user all the parameters, {', '.join(param_names)} \
        involved in the constraints and their physical meaning in an itemized list. 
        You can refer to the Model Summary I gave you to get the meaning of the parameters.\
             Avoid using any symbols of the parameters.  For example, answer to this step can be 
             "The following input data are involved in the constraints:
             P1. The molecular weight of a molecule A
             P2. the demand of customers 
             P3. the storage capacity"
    C. Tell the user they might want to change some data involved in {', '.join(param_names)} to make the model feasible, 
       but skip the parameters that have product with another variable in the constraints.\
       For this step, you should provide the user with an recommendation. To decide which parameters to recommend
        there is a rule of thumb you should consider:\
        In general, recommend parameters that can be easily change in the physical world. 
            For example, if I have the molecular weight of a molecule and the demand of customers in the parameters, 
            you should only recommend the demand of the customers to be changed because the molecular weight is a 
            physical property that cannot be changed.\
            
            DO NOT mention that "we don't recommend changing parameters a, b, c,.. etc because they have product with variables." \
            Use an explanation corresponding to the physical meaning of the parameters that makes them a good candidate. \
            An example answer would be
            "Based on my interpretation of your data, you might want to change the demand of the customers and expand 
            your storage capacity to make the model feasible."
            """
    status = resolve(model)
    if status == "optimal":
        return infer_feasibility(const_names, param_names, summary, gpt_model)
    else:
        explanation = get_completion_standalone(prompt, gpt_model)
    return explanation


def add_slack(param_names, model):
    """
    use <param_names> to add slack for ALL indices of the parameters
    """
    is_slack_added = {}  # indicator: is slack added to constraints?
    # define slack parameters
    for p in param_names:
        if eval("model." + p + ".is_indexed()"):
            is_slack_added[p] = {}
            for index in eval("model." + p + ".index_set()"):
                is_slack_added[p][index] = False
            exec("model.slack_pos_" + p + "=pe.Var(model." + p + ".index_set(), within=pe.NonNegativeReals)")
            exec("model.slack_neg_" + p + "=pe.Var(model." + p + ".index_set(), within=pe.NonNegativeReals)")

        else:
            is_slack_added[p] = False
            exec("model.slack_pos_" + p + "=pe.Var(within=pe.NonNegativeReals)")
            exec("model.slack_neg_" + p + "=pe.Var(within=pe.NonNegativeReals)")

    return is_slack_added

def generate_replacements(param_names, model):
    iis_param = []
    replacements_list = []
    for p_name in param_names:
        for idx in eval("model." + p_name + ".index_set()"):
            p_index = str(idx).replace("(", "[").replace(")", "]")

            if "[" and "]" in p_index:  # this happens when p is a parameter that has more than one index [idx1, idx2, ]
                p_name_index = p_name + p_index
            elif p_index == 'None':  # this happens when p is a parameter that doesn't have index
                p_name_index = p_name
            else:  # this happens when p is a parameter that has only one index [idx1]
                p_index = str([idx])
                p_name_index = p_name + p_index

            iis_param.append(p_name_index)
            expr_p = eval("model." + p_name_index)
            slack_var_pos = eval("model.slack_pos_" + p_name_index)
            slack_var_neg = eval("model.slack_neg_" + p_name_index)

            replacements = {id(expr_p): expr_p + slack_var_pos - slack_var_neg}
            replacements_list.append(replacements)
    return iis_param, replacements_list

def replace_const(replacements_list, model):
    """
    Replaces the constraints
    """
    const_list = []
    for const in model.component_objects(pe.Constraint):
        const_list.append(str(const))
    # const_list is a list containing all const_names in the model
    model.slack_iis = pe.ConstraintList()
    # replace each param in each const
    for const_name in const_list:
        consts = eval('model.' + const_name)
        for const_idx in consts:
            const = consts[const_idx]
            new_expr = clone_expression(const.expr)
            for replacements in replacements_list:
                new_expr = replace_expressions(new_expr, replacements)
            model.slack_iis.add(new_expr)
            const.deactivate()


def replace_obj(iis_param, model):
    # deactivate all the existing objectives
    objectives = model.component_objects(pe.Objective, active=True)
    for obj in objectives:
        obj.deactivate()

    # minimize the 1-norm of the slacks that are added
    new_obj = 0
    for p in iis_param:
        # other slack vars outside iis_param have been fixed to 0
        slack_var_pos = eval("model.slack_pos_" + p)
        slack_var_neg = eval("model.slack_neg_" + p)
        new_obj += slack_var_pos + slack_var_neg
    model.slack_obj = pe.Objective(expr=new_obj, sense=pe.minimize)


def resolve(model):
    opt = SolverFactory('gurobi')
    opt.options['nonConvex'] = 2
    opt.options['TimeLimit'] = 300  # 5min time limit
    results = opt.solve(model, tee=True)
    termination_condition = results.solver.termination_condition
    if termination_condition == "maxTimeLimit" and 'Upper bound' in results.Problem[0]:
        termination_condition = 'optimal'    
    return str(termination_condition)


def generate_slack_text(iis_param, model):
    text = "Model becomes feasible after the following change: "
    for p in iis_param:
        slack_var_pos = eval("model.slack_pos_" + p + ".value")
        slack_var_neg = eval("model.slack_neg_" + p + ".value")

        if slack_var_pos > 1e-5:
            text = text + f"increase {p} by {slack_var_pos} unit; "
        elif slack_var_neg > 1e-5:
            text = text + f"decrease {p} by {slack_var_neg} unit; "
    return text

def generate_sensitivity_text(dual_values, model):
    text = "The optimal value increases at the following rates: "
    for c in dual_values.keys():
        for values in dual_values[c]:
            text = text + f"{values[1]} for the constraint {c} at {values[0]}; "
    return text

def solve_the_model(param_names: list[str], param_names_aval, model) -> str:
    if all(param_name in param_names_aval for param_name in param_names):
        import copy
        model_copy = copy.deepcopy(model)
        is_slack_added = add_slack(param_names, model_copy)
        # all_const_in_model = find_const_in_model(model_copy)
        iis_param, replacements_list = generate_replacements(param_names, model_copy)
        replace_const(replacements_list, model_copy)
        replace_obj(iis_param, model_copy)
        termination_condition = resolve(model_copy)
        if termination_condition == 'optimal':
            out_text = generate_slack_text(iis_param, model_copy)
            flag = 'feasible'
        else:
            out_text = f"Changing {param_names} is not sufficient to make this model feasible, \n" \
                       f"Try other potential mutable parameters instead. \n"
            flag = 'infeasible'
    else:
        out_text = f"I can't help you change {param_names} " \
                   f"because they aren't valid mutable parameters in this model. \n"
        flag = 'invalid'
    return out_text, flag


def get_completion_from_messages_withfn(messages, gpt_model):
    functions = [
        {
            "name": "solve_the_model",
            "description": "Given the parameters to be changed, re-solve the model and report the extent of the changes",
            "parameters": {
                "type": "object",
                "properties": {
                    "param_names": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A parameter name"
                        },
                        "description": "List of parameter names to be changed in order to re-solve the model"
                    }
                },
                "required": ["param_names"]
            }
        },
        # {
        #     "name": "get_completion_detailed",
        #     "description": f"""This is an API call to an LLM that answer's the user's query. The LLM has access to the
        #     pyomo code file written in python, and using the comments given in the code, the LLM will come up with a simple
        #     real-world optimizatin problem that the given pyomo model is trying to solve. User will ask questions about it and the
        #     LLM will answer it. The LLM also has access to a json object `model_info`, which contains all the information about
        #     the parameters of the model, their dimension and their indices. The LLM can access the values of the model parameters at 
        #     the suitable indices as per the user's query and based on the story that theh LLM tells about what this optimization problem does. 
        #     User query can involve multiple paramters and each of them can possibly have different indices. """,
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "user_prompt": {
        #                 "type": "string",
        #                 "description": "The query asked by the user"
        #             }
        #         },
        #         "required": ["user_prompt"]
        #     }
        # },
        # {
        #     "name": "get_completion_with_context",
        #     "description": """This is an API call to an LLM that answer's the user's query. This LLM is acting like an infeasibility
        #     troubleshooter and is an expert in pyomo, linear and mixed integer programming problems. It knows the real world
        #     story about the optimization problem at hand, and knows all information about the model's solution, i.e.
        #     things like the infeasibility set, all the constraints, all the parameters and their physical meanings etc.
        #     But it doesn't know the exact dependencies of each of the model parameters. """,
        #     "parameters": {
        #         "type": "object",
        #         "properties": {
        #             "user_prompt": {
        #                 "type": "string",
        #                 "description": "The query asked by the user"
        #             }
        #         },
        #         "required": ["user_prompt"]
        #     }
        # },
        {
            "name": "sensitivity_analysis",
            "description": """Given the constraints to be changed, find the sensitivity coefficients for each of the constraints and report the values""",
            "parameters": {
                "type": "object",
                "properties": {
                    "constraint_names": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "A constraint name"
                        },
                        "description": "List of parameter names to be changed in order to re-solve the model"
                    }
                },
                "required": ["constraint_names"]
            }
        }
    ]
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        functions=functions,
        function_call='auto'
    )
    return response

def get_parameters_n_indices(model):
    params = {}
    for param in model.component_objects(pe.Param):
        is_indexed = param.is_indexed()
        dim = param.dim
        idx_set = [_ for _ in param.index_set()]
        params[str(param)] = {
            "is_indexed": is_indexed,
            "index_dim": dim,
            "index_set": idx_set
        }
    return params

def get_constraints_n_indices(model):
    constraints = {}
    for constraint in model.component_objects(pe.Constraint):
        is_indexed = constraint.is_indexed()
        dim = constraint.dim
        idx_set = [_ for _ in constraint.index_set()]
        constraints[constraint._name] = {
            "is_indexed": is_indexed,
            "index_dim": dim,
            "index_set": idx_set
        }
    return  constraints

def get_completion_detailed(user_prompt, model_info, PYOMO_CODE, gpt_model):
    messages = []
    system_message = {
        "role": "system",
        "content": f"""You are a Pyomo expert. You will be given a pyomo code file written in python, enclosed between
        triple back quotes. Your task is to understand the code and come up with a simple real-world optimization
        problem that the model is trying to solve. User will ask you questions about it and you should be able to answer them.
        You are also given a json object {model_info} which contains the parameters of the model. You should be able
        to access the values of the model parameters at the suitable indices when the user gives you a query based on
        the story that you tell about what this optimization problem does. User query can involve multiple paramters 
        and each of them can possibly have different indices. ONLY GENERATE WHAT IS ASKED. NO EXTRA TEXT.
        ```{PYOMO_CODE}```"""
    }
    messages.append(system_message)
    messages.append(user_prompt)
    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=messages,
        temperature=0,
    )
    return response


def get_completion_for_index_sensitivity(user_prompt, model_info, PYOMO_CODE, gpt_model, auto=None):
    functions = [
        {
            "name": "get_index_sensitivity",
            "description": "Get the actual index(s) of the model constraint(s) requested in natural language by the user, based on the json object",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "constraint": {
                                    "type": "string", 
                                    "description": "the constraint name"
                                },
                                "indices": {
                                    "type": "array", 
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": ["number", "string", "null"],
                                            "description": "Index corresponding to a dimension of the multi-dimensional index."
                                        },
                                        "description": "An index for the above constraint (as the index can be multi-dimensional)."
                                    }
                                }
                            },
                        },
                        "description": "The correct indices of the model constraint as per the user's query"
                    }
                },
                "required": ["index"]
            }
        }
    ]
    messages = []
    system_message = {
        "role": "system",
        "content": f"""You are a Pyomo expert. You will be given a pyomo code file written in python, enclosed between
        triple back quotes. Your task is to understand the code and come up with a simple real-world optimization
        problem that the model is trying to solve. User will ask you questions about it and you should be able to answer them.
        You are also given a json object {model_info} which contains the constraints of the model. You should be able
        to access the values of the model constraints at the suitable indices when the user gives you a query based on
        the story that you tell about what this optimization problem does. User query can involve multiple paramters 
        and each of them can possibly have different indices. ONLY GENERATE WHAT IS ASKED. NO EXTRA TEXT.
        ```{PYOMO_CODE}```"""
    }
    messages.append(system_message)
    messages.append(user_prompt)
   
    if auto:
        response = openai.ChatCompletion.create(
        model=gpt_model,
        messages = messages,
        functions = functions,
        function_call = "auto"
    )
    else:
        response = openai.ChatCompletion.create(
        model=gpt_model,
        messages = messages,
        functions = functions,
        function_call = {"name": "get_index_sensitivity"}
    )
    return response


def get_completion_for_index(user_prompt, model_info, PYOMO_CODE, gpt_model, auto=None):
    functions = [
        {
            "name": "get_index",
            "description": "Get the actual index(s) of the model parameter(s) requested in natural language by the user, based on the json object",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "parameter": {
                                    "type": "string", 
                                    "description": "the parameter name"
                                },
                                "indices": {
                                    "type": "array", 
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": ["number", "string", "null"],
                                            "description": "Index corresponding to a dimension of the multi-dimensional index."
                                        },
                                        "description": "An index for the above parameter (as the index can be multi-dimensional)."
                                    }
                                }
                            },
                        },
                        "description": "The correct indices of the model parameter as per the user's query"
                    }
                },
                "required": ["index"]
            }
        }
    ]
    messages = []
    system_message = {
        "role": "system",
        "content": f"""You are a Pyomo expert. You will be given a pyomo code file written in python, enclosed between
        triple back quotes. Your task is to understand the code and come up with a simple real-world optimization
        problem that the model is trying to solve. User will ask you questions about it and you should be able to answer them.
        You are also given a json object {model_info} which contains the parameters of the model. You should be able
        to access the values of the model parameters at the suitable indices when the user gives you a query based on
        the story that you tell about what this optimization problem does. User query can involve multiple paramters 
        and each of them can possibly have different indices. ONLY GENERATE WHAT IS ASKED. NO EXTRA TEXT.
        ```{PYOMO_CODE}```"""
    }
    messages.append(system_message)
    messages.append(user_prompt)
   
    if auto:
        response = openai.ChatCompletion.create(
        model=gpt_model,
        messages = messages,
        functions = functions,
        function_call = "auto"
    )
    else:
        response = openai.ChatCompletion.create(
        model=gpt_model,
        messages = messages,
        functions = functions,
        function_call = {"name": "get_index"}
    )
    return response

def gpt_function_call(ai_response, param_names_aval, model):
    fn_call = ai_response["choices"][0]["message"]["function_call"]
    fn_name = fn_call["name"]
    arguments = fn_call["arguments"]
    if fn_name == "solve_the_model":
        param_names = eval(arguments).get("param_names")
        return solve_the_model(param_names, param_names_aval, model), fn_name
    elif fn_name == "get_index":
        args = json.loads(arguments)
        return solve_the_model_indexed_new(args, model), "solve_the_model_indexed_new"
    elif fn_name == "get_index_sensitivity":
        args = json.loads(arguments)
        return solve_sensitivity_indexed(args, model), "solve_sensitivity_indexed"
    else:
        return

def add_slack_indexed_new(objs, model):
    is_slack_added = {}  # indicator: is slack added to constraints?
    # define slack parameters
    for i in objs:
        param_name = i['parameter']
        indices = i['indices']
        if eval(f"model.{param_name}.is_indexed()"):
            for index in indices:
                pass
                # is_slack_added[param_name][index] = False
            exec(f"model.slack_pos_{param_name} = pe.Var({indices}, within=pe.NonNegativeReals)")
            exec(f"model.slack_neg_{param_name} = pe.Var({indices}, within=pe.NonNegativeReals)")
        else: 
            # is_slack_added[param_name] = False
            exec("model.slack_pos_" + param_name + "=pe.Var(within=pe.NonNegativeReals)")
            exec("model.slack_neg_" + param_name + "=pe.Var(within=pe.NonNegativeReals)")

    return is_slack_added

def generate_replacements_indexed_new(objs, model):
    iis_param = []
    replacements_list = []
    for i in objs:
        p_name = i['parameter']
        indices = i['indices']
        for idx in indices:
            idx = str(idx)
            if "[" and "]" in idx:
                p_name_index = p_name + idx
            else:
                p_name_index = p_name
            
            iis_param.append(p_name_index)
            expr_p = eval(f"model.{p_name_index}")
            slack_var_pos = eval(f"model.slack_pos_{p_name_index}")
            slack_var_neg = eval(f"model.slack_neg_{p_name_index}")

            replacements = {id(expr_p): expr_p + slack_var_pos - slack_var_neg}
            replacements_list.append(replacements)
    return iis_param, replacements_list

def solve_sensitivity_indexed(args, model):
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    # model_copy = model.clone()
    dual_values = {}
    model.dual = pe.Suffix(direction=pe.Suffix.IMPORT)
    solver = SolverFactory("gurobi")
    # solver.solve(model_copy, tee=False)
    for var in model.component_objects(pe.Var):
        for index in var.index_set():
            var[index].domain = pe.NonNegativeReals
    results = solver.solve(model, tee=False)
    # import pdb
    # pdb.set_trace()
    termination_condition = results.solver.termination_condition
    if termination_condition == "maxTimeLimit" and 'Upper bound' in results.Problem[0]:
        termination_condition = 'optimal'
    
    if termination_condition == "optimal":
        for arg in args['index']:
            c_name = arg['constraint']
            indices = arg['indices']
            dual_values[c_name] = []
            for idx in indices:
                idx = str(idx)
                c_name_index = c_name + idx
                dual_value = eval(f"model.dual[model.{c_name_index}]")
                dual_values[c_name].append((idx, dual_value))
        out_text = generate_sensitivity_text(dual_values, model)
        flag = "feasible"
        return out_text, flag
    else:
        out_text = f"Since the model is infeasible, it is not possile to answer the above question\n"
        flag = "infeasible"
        return out_text, flag

def solve_the_model_indexed_new(args, model):
    model_copy = model.clone()
    is_slack_added = add_slack_indexed_new(args['index'], model_copy)
    iis_param, replacements_list = generate_replacements_indexed_new(args['index'], model_copy)
    replace_const(replacements_list, model_copy)
    replace_obj(iis_param, model_copy)
    termination_condition = resolve(model_copy)
    if termination_condition == 'optimal':
        out_text = generate_slack_text(iis_param, model_copy)
        flag = 'feasible'
    else:
        out_text = f"Changing {args['index']} is not sufficient to make this model feasible, \n" \
                    f"Try other potential mutable parameters instead. \n"
        flag = 'infeasible'
    return out_text, flag

def evaluate_gpt_response(question, answer, model_info, PYOMO_CODE, gpt_model):
    evaluation_prompt = []
    evaluation_prompt.append({
        "role": "system",
        "content": """You are an expert that can reason and determine if your junior AI assistant says it
        knows what is being asked or not. ANSWER IN YES/NO. You are also given a json object {model_info} and the optimization model in pyomo enclosed in triple back quotes. You should be able
        to access the values of the model parameters at the suitable indices as per the user query. ONLY GENERATE WHAT IS ASKED. NO EXTRA TEXT.
        ```{PYOMO_CODE}```"""
    })
    evaluation_prompt.append({
        "role": "user",
        "content": f"""{question}"""
    })
    evaluation_prompt.append({
        "role": "assistant",
        "content": f"{answer}"
    })
    evaluation_prompt.append({
        "role": "user",
        "content": "Did the junior assistant answer correctly what is being asked? If you think its answer is out of scope for you, then say YES."
    })
    response = openai.ChatCompletion.create(
    model=gpt_model,
    messages=evaluation_prompt,
    temperature=0)
    return response["choices"][0]["message"]["content"]

def classify_question(question, answer, model_info, gpt_model):
    evaluation_prompt = []
    evaluation_prompt.append({
        "role": "system",
        "content": f"""You are an expert in optimization models and pyomo. You will be given a user query, and an AI assistant's response.
        You have to determine to whom this query is to be forwarded. There are two experts whose description
        is given as follows:
        
        1. Expert 1: He is an expert in pyomo and infeasibility troubleshooting. He knows the real world
        story about the optimization problem, and he knows all information about the model's solution, i.e.
        things like the infeasibility set, all the constraints, all the parameters and their physical meanings etc.
        But he doesn't know the exact dependencies of each of the model parameters. For example, he knows that
        a particular parameter depends on the number of tasks, but he doesn't know what those tasks are.

        2. Expert 2: He is a python programming expert. He has access to the pyomo code of the model (which has detailed doc string and comments),
        and he also has the fine-grained information of the optimization model as a json object. For example, he knows the values
        of each and every python variable, the indices of each and every model parameter etc. But he doesn't know the physical meanings
        of them.
        

        You have access to a json object {model_info} which has all the pyomo model details. Verify the answer of the AI model with this info, and if it is incorret,
        forward the query to expert 2.
        As an expert, you have to decide to whom the user query is to be forwarded. Answer `1` if you want to forward it to Expert 1.
        Otherwise answer `2`. GENERATE ONLY WHAT IS ASKED, AND NOTHING ELSE!! Please take your time to properly think it out"""
    })
    evaluation_prompt.append(question)
    answer['role'] = "user"
    answer['content'] = f"""The AI assistant says: {answer['content']}"""
    evaluation_prompt.append(answer)
    response = openai.ChatCompletion.create(
    model=gpt_model,
    messages=evaluation_prompt,
    temperature=0)
    return response["choices"][0]["message"]["content"]