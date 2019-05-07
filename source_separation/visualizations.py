from datetime import datetime
from time import perf_counter as clock
import matplotlib.pyplot as plt
import numpy as np
import webbrowser
import visdom
from source_separation.data_objects import get_instrument_name


class Visualizations:
    def __init__(self, env_name=None, averaging_window=25, auto_open_browser=True):
        # Set the environement name with the current time
        now = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = now if env_name is None else "%s (%s)" % (env_name, now)
        # FIXME
        self.env_name = env_name
        
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
        self.waveforms_win = None
        self.audio_wins = []
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
        
        # # Plot the learning rate
        # self.lr_win = self.vis.line(
        #     [lr],
        #     [step],
        #     win=self.lr_win,
        #     update="append" if self.lr_win else None,
        #     opts=dict(
        #         xlabel="Step",
        #         ylabel="Learning rate",
        #         ytype="log",
        #         title="Learning rate"
        #     )
        # )
        
    def draw_waveform(self, y_pred, y_true, instruments):
        # Waveform plots
        fig, axs = plt.subplots(len(y_pred), 2)
        for i, y in enumerate([y_true, y_pred]):
            for j, yj in enumerate(y):
                ax = axs[j, i]
                ax.plot(yj)
                ax.set_ylim(-1.05, 1.05)
        self.waveforms_win = self.vis.matplot(fig, win=self.waveforms_win)
        plt.close(fig)
        
        # Audios
        self.audio_wins.extend([None] * (len(y_pred) - len(self.audio_wins)))
        for i in range(len(y_pred)):
            # This is a stupid fix to prevent visdom from normalizing the audio
            y_pred[i][-1] = 1
            y_pred[i][-2] = -1
            self.audio_wins[i] = self.vis.audio(y_pred[i], win=self.audio_wins[i], opts=dict(
                title=get_instrument_name(instruments[i])
            ))
        
    def save(self):
        self.vis.save([self.env_name])
        
        