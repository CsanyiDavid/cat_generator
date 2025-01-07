from torchview import draw_graph

def draw_model_graph(model, output_name):
    model_graph = draw_graph(model, input_size=(1,1,64,64), expand_nested=True)
    model_graph.visual_graph.format = "png"
    model_graph.visual_graph.render("./../train_logs/" + output_name, cleanup=True)