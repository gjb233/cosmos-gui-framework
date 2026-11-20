"""Split the installed original MAI forward at its decoder boundary.
Uses the original function globals and statements; shared source is untouched.
"""
import ast
import copy
import inspect
import textwrap

def build_split_forward(network_class):
    original = network_class.forward
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    matches = []
    for i,node in enumerate(function.body):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
            call=node.value.func
            if isinstance(call,ast.Attribute) and isinstance(call.value,ast.Name) and call.value.id=='self' and call.attr=='language_model':
                matches.append(i)
    if len(matches)!=1:
        raise ValueError("Expected exactly one original decoder boundary")
    index=matches[0]
    prep=copy.deepcopy(function)
    prep.name='prepare_original_fm'
    prep.decorator_list=[]
    prep.body=ast.parse("if memory is not None or hasattr(packed_seq, 'gui_ar_decode_request'): raise ValueError('Native training only')").body + prep.body[:index]
    prep.body.extend(ast.parse("""
if memory is not None or hasattr(packed_seq, 'gui_ar_decode_request'):
    raise ValueError('Native training only: no inference memory/decode')
if natten_metadata_list is not None:
    raise NotImplementedError('NATTEN stage metadata not migrated')
return dict(input_pack=input_pack, attention_meta=attention_meta,
            packed_position_ids=packed_position_ids,
            packed_seq=packed_seq, original_latent_shapes=original_latent_shapes,
            original_latent_shapes_lidar=original_latent_shapes_lidar,
            sequence_shard_parallel_dims=sequence_shard_parallel_dims,
            correct_cp_gradients=correct_cp_gradients)
""").body)
    finish=ast.parse("def finish_original_fm(self, packed_outputs, context):\n pass").body[0]
    names=['packed_seq','original_latent_shapes','original_latent_shapes_lidar',
           'sequence_shard_parallel_dims','correct_cp_gradients']
    finish.body=ast.parse("\n".join(n+" = context["+repr(n)+"]" for n in names)+"\nlbl_metadata = {}").body
    for node in function.body[index+1:]:
        # Native causal AR is run through the shared pipeline stages, so do not
        # invoke the second full decoder forward in the original tail.
        if isinstance(node,ast.If) and any(isinstance(n,ast.Name) and n.id=='hybrid_ar_forward' for n in ast.walk(node)):
            continue
        finish.body.append(copy.deepcopy(node))
    module=ast.Module(body=[prep,finish],type_ignores=[])
    ast.fix_missing_locations(module)
    namespace=dict(original.__globals__)
    exec(compile(module,'<isolated-mai-forward-split>','exec'),namespace)
    return namespace[prep.name],namespace[finish.name],ast.unparse(module)
