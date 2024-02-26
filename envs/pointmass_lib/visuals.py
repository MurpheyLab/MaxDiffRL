#!/usr/bin/env python3

# Graphics-related
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D
import seaborn as sns

class PM_Viewer(object):
    def __init__(self,args,display=True):

        dist = args.env['boundary_distance']
        if dist <= 0:
            dist =  args.env['start_distance']*10.
        inset = args.env['start_distance']*1.5
        self.boundary = (-dist,dist)
        self.boundary2 = (-inset,inset)

        # title params
        self.method = args.method
        self.beta = args.beta
        self.alpha = args.alpha
        self.horizon = args.horizon
        self.seed = args.seed

        self.display = display

        if self.display and 'TkAgg' in matplotlib.rcsetup.interactive_bk: 
            matplotlib.use('TkAgg')

        self.build_fig(dist,inset)

        if self.display:
            plt.ion()
            plt.show(block=False)

    def update_title(self):
        title = 'method = {}, beta = {}, alpha = {}, horizon = {}, seed = {}'.format(self.method, self.beta, self.alpha, self.horizon, self.seed)
        self.fig.suptitle(title)

    def build_fig(self,dist,inset):

        sns.set_palette("pastel")
        plt.rcParams.update({'font.size': 7,'figure.figsize':(7.5 , 3.44)})

        self.fig, axs = plt.subplots(1, 2)

        # make fixed axes
        x_axis = np.linspace(-1,1,1000)[:, None]
        y_axis = np.linspace(-1,1,1000)[None, :]
        arr = np.sqrt(x_axis ** 2 + y_axis ** 2)
        for ax, boundary in zip(axs,[self.boundary,self.boundary2]):
            for lab,style in zip(['Start','End','Goal'],['wo','rs','wx']):
                ax.plot(np.NaN, np.NaN, style,label=lab)
            im = ax.imshow(arr,cmap='gist_gray',extent=(-dist,dist,-dist,dist), interpolation='none',norm=matplotlib.colors.PowerNorm(gamma=0.5))
            ax.set_title(' ')
            ax.set_aspect('equal')
            ax.set(xlim=boundary,ylim=boundary)

        inset_patch = plt.Rectangle((-inset,-inset),inset*2,inset*2,facecolor='none',edgecolor='magenta')
        axs[0].add_patch(inset_patch)
        for coords1,coords2 in zip([(inset,-inset),(inset,inset)],[(-inset,-inset),(-inset,inset)]):
            con = ConnectionPatch(xyA=coords1, xyB=coords2, coordsA="data", coordsB="data", axesA=axs[0], axesB=axs[1],color='magenta')
            axs[1].add_artist(con)
        axs[1].tick_params(color='magenta', labelcolor='magenta')
        axs[0].legend(loc="upper left",facecolor='black',labelcolor='white',prop=dict(weight='bold'))

        # make second axis
        self.ax = axs[0].twinx()
        self.ax2 = axs[1].twinx()
        plt.setp(self.ax2.spines.values(), color='magenta');
        for ax in [self.ax,self.ax2]:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        # shift everything over to make room for legend
        for ax in [*axs,self.ax,self.ax2]:
            ax_box = ax.get_position()
            ax_box.x0 = ax_box.x0 - 0.015
            ax_box.x1 = ax_box.x1 - 0.015
            ax.set_position(ax_box)

        # set title
        self.update_title()

    def render(self,states,ep_num,tensor=False,clear_fig=True,ncol=1):
        if tensor:
            xy = torch.stack(states).cpu().numpy()
        else:
            xy = np.array(states)
        color = None
        if isinstance(ep_num,int):
            if ep_num % 10 == 0  and clear_fig:
                for ax in [self.ax,self.ax2]:
                    ax.clear()
            label = '{}'.format(ep_num)
        else:
            label = ep_num
            color = 'tab:brown'
        for ax,clip_on,boundary in zip([self.ax,self.ax2],[False,True],[self.boundary,self.boundary2]):
            ax.plot(xy[:,0],xy[:,1],'-+',color=color,label=label)
            ax.plot(0,0,'wx')
            ax.plot(xy[0,0],xy[0,1],'wo')
            ax.plot(xy[-1,0],xy[-1,1],'rs')
            # set lims
            ax.set_aspect('equal')
            ax.set(xlim=boundary,ylim=boundary)
        legend = self.ax2.legend(bbox_to_anchor=(1.05,0.95), loc="upper left",ncol=ncol,
                        facecolor='black',labelcolor='linecolor',prop=dict(weight='bold'),title='Path')
        plt.setp(legend.get_title(), color='white')
        plt.setp(legend.get_title(), weight='bold')

        if self.display:
            try: # doesn't bring figure to front of all screens, but doesn't work for all backends
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except:
                plt.draw()
                plt.pause(1e-4)

    def save(self,path):
        plt.savefig(path) 

    def close(self):
        plt.close(self.fig)

class PM_Viewer_plain(object):
    def __init__(self,args,display=True,minimal=True):
        dist = args.env['boundary_distance']
        self.print_boundary = True
        if dist <= 0:
            dist =  args.env['start_distance']*10.
            self.print_boundary = False
        inset = args.env['start_distance']*1.5
        self.boundary = (-dist,dist)
        self.boundary2 = (-inset,inset)
        self.minimal = minimal

        # title params
        self.method = args.method
        self.beta = args.beta
        self.alpha = args.alpha
        self.horizon = args.horizon
        self.seed = args.seed

        self.display = display

        if self.display and 'TkAgg' in matplotlib.rcsetup.interactive_bk: 
            matplotlib.use('TkAgg')

        self.build_fig(dist,inset)

        if self.display:
            plt.ion()
            plt.show(block=False)

    def update_title(self):
        title = 'method = {}, beta = {}, alpha = {},\nhorizon = {}, seed = {}'.format(self.method, self.beta, self.alpha, self.horizon, self.seed)
        if self.print_boundary:
            title = title + ', boundary = {}'.format(self.boundary[-1])
        self.fig.suptitle(title)

    def build_extra_legend(self):
        legend_elements = [Line2D([0], [0], marker=style, color=color, label=lab,markerfacecolor=color, markersize=5,linewidth=0) for lab,color,style in zip(['Start','End','Goal'],['k','r','k'],['o','s','x'])]
        extra_legend = plt.legend(handles=legend_elements, loc='upper left')
        self.ax.add_artist(extra_legend)

    def build_fig(self,dist,inset):

        sns.set_palette("deep")
        plt.rcParams.update({'font.size': 7,'figure.figsize':(3.75, 3.44)})

        self.fig, self.ax = plt.subplots(1, 1)

        # shift everything over to make room for legend
        if not self.minimal:
            ax_box = self.ax.get_position()
            ax_box.x0 = ax_box.x0 - 0.04
            ax_box.x1 = ax_box.x1 - 0.04
            self.ax.set_position(ax_box)

        # set title
        self.update_title()
        if not self.minimal:
            self.build_extra_legend()

    def render(self,states,ep_num,tensor=False,clear_fig=True,ncol=1):
        if tensor:
            xy = torch.stack(states).cpu().numpy()
        else:
            xy = np.array(states)
        color = None
        if isinstance(ep_num,int):
            if ep_num % 10 == 0  and clear_fig:
                self.ax.clear()
                self.build_extra_legend()
            label = '{}'.format(ep_num)
            lw = 1
        else:
            label = ep_num
            color = 'tab:brown'
            lw = '2'
        if self.minimal:
            self.ax.plot(xy[::2,0],xy[::2,1],color=color,label=label,linewidth=2)
        else:
            self.ax.plot(xy[:,0],xy[:,1],'-+',color=color,label=label,linewidth=lw)
            self.ax.plot(0,0,'kx')
            self.ax.plot(xy[0,0],xy[0,1],'ko') 
            self.ax.plot(xy[-1,0],xy[-1,1],'rs') 
            legend = self.ax.legend(bbox_to_anchor=(1.02,0.95), loc="upper left",ncol=ncol,title='Path')
        # set lims
        self.ax.set_aspect('equal','box')
        self.ax.set(xlim=self.boundary2,ylim=self.boundary2)

        if self.display:
            try: # doesn't bring figure to front of all screens, but doesn't work for all backends
                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            except:
                plt.draw()
                plt.pause(1e-4)

    def save(self,path):
        plt.savefig(path) 

    def close(self):
        plt.close(self.fig)
