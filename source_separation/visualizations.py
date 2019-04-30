from datetime import datetime
from time import perf_counter as clock
import matplotlib.pyplot as plt
import numpy as np
import webbrowser
import visdom


class Visualizations:
    def __init__(self, env_name=None, averaging_window=25, auto_open_browser=True):
        # Set the environement name with the current time
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = now if env_name is None else "%s (%s)" % (env_name, now)
        
        # Connect to a visdom instance
        try:
            self.vis = visdom.Visdom(env=self.env_name, raise_exceptions=True)
        except ConnectionError:
            raise Exception("No visdom server detected. Run the command \"visdom\" in your CLI to "
                            "start it.")
        if auto_open_browser:
            webbrowser.open("http://localhost:8097/env/" + self.env_name)

        self.averaging_window = averaging_window
        self.loss_win = None
        self.lr_win = None
        self.implementation_win = None
        self.projection_win = None
        self.loss_buffer = []
        self.implementation_string = ""
        self.last_step = -1
        self.last_update_timestamp = -1
        self.mean_time_per_step = -1
        
    def log_params(self, params, title):
        param_string = "<b>%s</b>:<br>" % title
        for param_name, param_value in params.items():
            param_string += "\t%s: %s<br>" % (param_name, param_value)
        self.vis.text(param_string, opts={"title": title})
        
    def log_implementation(self, params):
        implementation_string = ""
        for param, value in params.items():
            implementation_string += "<b>%s</b>: %s\n" % (param, value)
            implementation_string = implementation_string.replace("\n", "<br>")
        self.implementation_string = implementation_string
        self.implementation_win = self.vis.text(
            implementation_string, 
            opts={"title": "Training implementation"}
        )

    def update(self, loss, lr, step):
        # Plot the loss
        self.loss_buffer.append(loss)
        if len(self.loss_buffer) > self.averaging_window:
            del self.loss_buffer[0]
        self.loss_win = self.vis.line(
            [[loss, np.mean(self.loss_buffer)]],
            [[step, step]],
            win=self.loss_win,
            update="append" if self.loss_win else None,
            opts=dict(
                legend=["Loss", "Avg. loss"],
                xlabel="Step",
                ylabel="Loss",
                title="Loss",
            )
        )
        
        # Plot the learning rate
        self.lr_win = self.vis.line(
            [lr],
            [step],
            win=self.lr_win,
            update="append" if self.lr_win else None,
            opts=dict(
                xlabel="Step",
                ylabel="Learning rate",
                ytype="log",
                title="Learning rate"
            )
        )
        
        # now = clock()
        # if self.last_step != -1 and self.implementation_win is not None:
        #     time_per_step = (now - self.last_update_timestamp) / (step - self.last_step)
        #     if self.mean_time_per_step == -1:
        #         self.mean_time_per_step = time_per_step
        #     else:
        #         self.mean_time_per_step = self.mean_time_per_step * 0.9 + time_per_step * 0.1
        #     time_string = "<b>Mean time per step</b>: %dms" % int(1000 * self.mean_time_per_step)
        #     time_string += "<br><b>Last step time</b>: %dms" % int(1000 * time_per_step)
        #     self.vis.text(
        #         self.implementation_string + time_string, 
        #         win=self.implementation_win,
        #         opts={"title": "Training implementation"},
        #     )
        #     print("Step %6d   Loss: %.4f   EER: %.4f   LR: %g   Mean step time: %5dms   "
        #           "Last step time: %5dms" %
        #           (step, self.loss_buffer, self.eer_exp, lr, int(1000 * self.mean_time_per_step),
        #            int(1000 * time_per_step)))
        #     
        # self.last_step = step
        # self.last_update_timestamp = now
        
    # def draw_projections(self, embeds, utterances_per_speaker, step, out_fpath=None,
    #                      max_speakers=10):
    #     max_speakers = min(max_speakers, len(colormap))
    #     embeds = embeds[:max_speakers * utterances_per_speaker]
    #     
    #     n_speakers = len(embeds) // utterances_per_speaker
    #     ground_truth = np.repeat(np.arange(n_speakers), utterances_per_speaker)
    #     colors = [colormap[i] for i in ground_truth]
    #     
    #     reducer = umap.UMAP()
    #     projected = reducer.fit_transform(embeds)
    #     plt.scatter(projected[:, 0], projected[:, 1], c=colors)
    #     plt.gca().set_aspect("equal", "datalim")
    #     plt.title("UMAP projection (step %d)" % step)
    #     self.projection_win = self.vis.matplot(plt, win=self.projection_win)
    #     if out_fpath is not None:
    #         plt.savefig(out_fpath)
    #     plt.clf()
        
    # def save(self):
    #     self.vis.save([self.env_name])
        