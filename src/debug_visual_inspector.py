import plotly.express as px
import plotly.offline as pyo

# ---------------
# WORKING
# ---------------
# class_index = 0
# figure = px.imshow(batch_labels[-1,...,class_index].detach().cpu(), animation_frame = -1, title = "labels")
# pyo.plot(figure, filename='figure_labels.html')

# figure = px.imshow(probe_output[-1,...,class_index].detach().cpu(), animation_frame = 0, title = "probe_output")
# pyo.plot(figure, filename='figure_probe.html')

# diff = probe_output[-1,...,class_index].detach().cpu()-batch_labels[-1,...,class_index].detach().cpu()
# figure = px.imshow(diff, animation_frame = 0, title = "Diff (out-labels)", range_color=[-1, 1])
# pyo.plot(figure, filename='figure_diff.html')

#------------------



# ------------
# From train_probes.py
# ------------

                            # figure = px.imshow(batch_labels[-1,...,0].detach().cpu(), animation_frame = 0, title = "labels")
                            # pyo.plot(figure, filename=self.visuals_prefix+'figure_labels.html')

                            # figure = px.imshow(probe_output[-1,...,0].detach().cpu(), animation_frame = 0, title = "probe_output")
                            # pyo.plot(figure, filename=self.visuals_prefix+'figure_probe.html')

                            # diff = probe_output[-1,...,0].detach().cpu()-batch_labels[-1,...,0].detach().cpu()
                            # figure = px.imshow(diff, animation_frame = 0, title = "Diff (out-labels)", range_color=[-1, 1])
                            # pyo.plot(figure, filename=self.visuals_prefix+'figure_diff.html')


import torch
batch_labels = torch.tensor([])

class_index = 0

def make_plot(tensor,title):
    positions = tensor.shape[0]
    tensor = tensor.permute(0,3,1,2)
    flat_tens = tensor.reshape(positions,-1,8)
    figure = px.imshow(flat_tens.detach().cpu(), animation_frame = 0, title = "labels")
    return pyo.plot(figure, filename=f'figure_{title}.html')

make_plot(batch_labels[-1],'labels')


.permute(0,1,3,2)