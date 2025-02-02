import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from matplotlib import pyplot as plt
import torch


class viz():
    def __init__(self):
        pass
    def Abunds(self, pred=None,
                ground=None,
                nr=None,
                nc=None,
                colorscale = "rainbow",
                thetitle='title',
                savepath=None):
        ''' plots abundance maps, should be P by N '''
        P = pred.shape[0]  # number of endmembers
        N = pred.shape[1]  # number of pixels
        A_cube_pred = pred.T.reshape((nc, nr, P))
        A_cube_ground = ground.T.reshape((nc, nr, P))

        fig = make_subplots(rows=2, cols=P)

        # Add subplots for predicted abundance maps
        for i in range(P):
            fig.add_trace(go.Heatmap(z=A_cube_pred[:, :, i].T, colorscale=colorscale, zmin=0, zmax=1, showscale=True,
                        name=f"heat_{i}"),
                        row=1, col=i+1)

        # Add color bar for predicted abundance maps
        fig.update_layout(coloraxis_colorbar=dict(title='Predicted'))

        # Add subplots for ground truth abundance maps
        for i in range(P):
            fig.add_trace(go.Heatmap(z=A_cube_ground[:, :, i].T, colorscale=colorscale, zmin=0, zmax=1, showscale=True, colorbar=dict(),
                        name=f"heat_{i}"),
                        row=2, col=i+1)

        # Add color bar for ground truth abundance maps
        fig.update_layout(coloraxis_colorbar=dict(title='Ground Truth'))

        # Update y-axis titles for each row
        fig.update_yaxes(title_text="Predicted", row=1, col=1)
        fig.update_yaxes(title_text="Ground Truth", row=2, col=1)

        fig.update_layout(title=thetitle, title_font_size=18)#,# height=500, width=800)

        if savepath is not None:  # save a figure is specified
            fig.write_image(savepath, format='pdf')

        fig.show()
        return fig

    def plotEMs(self, M, thetitle='title', savepath=None):
        P = M.shape[1] # number of endmembers
        L = M.shape[0] # number of bands

        fig = go.Figure()

        for i in range(P):
            fig.add_trace(go.Scatter(x=torch.linspace(1, L, L), y=M[:, i], mode='lines', name=f'Endmember {i+1}'))

        fig.update_layout(title=thetitle, title_font_size=18, xaxis_title='Band', yaxis_title='Reflectance')

        if savepath is not None:
            fig.write_image(savepath, format='pdf')

        fig.show()
        return fig
    


def reshape_fortran(x, shape):
    ''' perform a reshape in Matlab/Fortran style '''
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def plotImage(dataY, nr, nc, L=None):
    '''plots an image, a few bands (L by N)'''
    # if isinstance(dataY, list):
        # Y = torch.zeros((nr,nc,L))
    if L == None:
        L = dataY.shape[0]
    plt.figure()
    Yim = torch.reshape(dataY.T, (nc,nr,L))
    plt.imshow(Yim[:,:,[10, 20, 30]])
    plt.show()


def plotEMs(M, thetitle='title'):
    P = M.shape[1] # number of endmembers
    L = M.shape[0] # number of bands
    fig = plt.figure()
    for i in range(P):
        plt.plot(torch.linspace(1,L,L), M[:,i])
    plt.title(thetitle, fontsize=12)
    plt.show()


def plotAbunds(A, nr, nc, thetitle='title', savepath=None):
    ''' plots abundance maps, should be P by N '''
    P = A.shape[0] # number of endmembers
    N = A.shape[1] # number of pixels
    A_cube = torch.reshape(A.T, (nc,nr,P))
    fig, axs = plt.subplots(1, P)
    for i in range(P):
        axs[i].imshow(A_cube[:,:,i].T, cmap='jet', vmin=0, vmax=1) #cmap='gray'
        axs[i].axis('off')
    # plt.axis('off')
    # axs[P-1].colorbar()
    # fig = plt.figure()
    # # plt.imshow(temp)
    # plt.imshow(A_cube[:,:,0], cmap='gray', vmin=0, vmax=1)
    # plt.colorbar()
    plt.title(thetitle, fontsize=12)
    if savepath is not None: # save a figure is specified
        plt.savefig(savepath, dpi=300, format='pdf')
    plt.show()


def show_ground_truth(A_true, Mgt_avg, nr, nc):
    plotEMs(Mgt_avg, thetitle='ground truth')
    plotAbunds(A_true, nr=nr, nc=nc, thetitle='ground truth')


def compute_metrics(t_true, t_est):
    RMSE = torch.sqrt(torch.sum((t_true-t_est)**2)/t_true.shape.numel())
    NRSME = torch.sqrt(torch.sum((t_true-t_est)**2)/torch.sum(t_true**2))
    return RMSE, NRSME

