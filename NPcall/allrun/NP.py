'''import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])

install('plotly')
'''
import os
import platform
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import sys


def quantile_norm(data):
    temp = data.argsort(axis = 0)
    rank = np.empty_like(temp)
    for i in range(data.shape[1]):
        rank[temp[:,i],i] = np.arange(data.shape[0])
    return np.sort(data, axis = 0).mean(axis = 1)[rank]

#%%
## LDA and scatter plots #####################################################################################################################
def generate_LDA_ann_text(res, All_bg, datatype, time = 3):
    npp = res[['mean_2', 'mean_4']].values
    lda_label = LinearDiscriminantAnalysis()
    lda_label.fit(np.log2(npp), res['label'].values)
    prediction_label = lda_label.predict(np.log2(npp))
    pred_corr_label = prediction_label == res['label'].values
    cont_table = pd.crosstab(prediction_label.astype(np.int32), res['label'].values.astype(np.int32), 
                               rownames = ['Prediction'], colnames = ['Truth'])
    print(datatype)
    print(cont_table)
    
    precision = cont_table.iloc[0,0] / (cont_table.iloc[0,0] + cont_table.iloc[0,1])
    recall = cont_table.iloc[0,0] / (cont_table.iloc[0,0] + cont_table.iloc[1,0])
    # Fisher score
    label2 = np.log2(res[res.label == 2].loc[:, ['mean_2', 'mean_4']].values).T
    label4 = np.log2(res[res.label == 4].loc[:, ['mean_2', 'mean_4']].values).T
    S1 = np.cov(label2) * (label2.shape[1] - 1)
    S2 = np.cov(label4) * (label4.shape[1] - 1)
    Sw = S1 + S2
    mean1 = np.mean(label2, axis = 1)
    mean2 = np.mean(label4, axis = 1)
    diff = (mean1 - mean2).reshape(-1, 1)
    Sb = np.dot(diff, diff.T)
    v = np.dot(np.linalg.inv(Sw), (mean1 - mean2)).reshape(-1, 1)
    J = (np.dot(np.dot(v.T, Sb), v)[0][0]) / (np.dot(np.dot(v.T, Sw), v)[0][0])
    
    std = np.std(np.log2(All_bg.loc[:, ['mean_2', 'mean_4']].values), axis = 0)
    mean = np.mean(np.log2(All_bg.loc[:, ['mean_2', 'mean_4']].values), axis = 0)
    critical = mean + time*std

    ann_text = ['● Accuracy: {:.2f}%'.format((sum(pred_corr_label) / len(res))*100) +
                '<br>● Classification error: {:.2f}%'.format((1 - (sum(pred_corr_label) / len(res)))*100) +
                '<br>● Precision: {:.2f}%'.format((precision)*100) +
                '<br>● Recall: {:.2f}%'.format((recall)*100) +
                '<br>● Fisher Score: {:.2e}<br><br>'.format(J) +
                '<br>● Critical value (3·SD):' + 
                '<br>&nbsp; &nbsp; Channel 2: {:.3f}'.format(critical[0]) +
                '<br>&nbsp; &nbsp; Channel 4: {:.3f}'.format(critical[1])]
    return critical, ann_text, pred_corr_label
#%%
def generate_scatter_trace(trace_name, all_scatter_data_x, all_scatter_data_y, 
                           modes, names, markers, lines, legendgroups, showlegends):
    for par in zip(trace_name, all_scatter_data_x, all_scatter_data_y, modes, names, 
                   markers, lines, legendgroups, showlegends):
        globals()[par[0]] = go.Scatter(
                                x = par[1],
                                y = par[2],
                                mode = par[3],
                                name = par[4],
                                marker = par[5],
                                line = par[6],
                                legendgroup = par[7],
                                showlegend = par[8]
                                )
#%%
def generate_scatter_layout(layout_name, scatter_titles, x_titles, y_titles, 
                            data_min, data_max, ann_subtitle, ann_text, file_formats):
        
    for par in zip(layout_name, scatter_titles, x_titles, y_titles, 
                   data_min, data_max, ann_subtitle, ann_text, file_formats):

        if (par[8] == 'html'):
            title_x = 0.05; title_font_size = 14; 
            xaxis_title_font_size = 8; xaxis_domain = [0, 0.47];
            yaxis_title_font_size = 8; yaxis_domain = [0, 1];
            legend_x = 0.5; legend_y = 0.8; legend_bgcolor = 'rgba(255, 255, 255, 1)';
            ann_subtitle_x = 0.7; ann_subtitle_y = 1; ann_subtitle_font_size = 14;
            ann_text_x = 0.72; ann_text_y = 0.9; ann_text_font_size = 12
    
        elif (par[8] == 'pdf'):
            title_x = 0.09; title_font_size = 10;
            xaxis_title_font_size = 8; xaxis_domain = [0, 0.7];
            yaxis_title_font_size = 8; yaxis_domain = [0, 1];
            legend_x = 0.5; legend_y = 0.8; legend_bgcolor = 'rgba(255, 255, 255, 1)';
            ann_subtitle_x = 0.75; ann_subtitle_y = 1; ann_subtitle_font_size = 10;
            ann_text_x = 0.77; ann_text_y = 0.9; ann_text_font_size = 8
    
        globals()[par[0]] = go.Layout(
                                autosize = True,
                                title = dict(text = par[1],
                                             x = title_x,
                                             font = dict(size = title_font_size)),
                                xaxis = dict(title = dict(text = par[2],
                                                          font = dict(size = xaxis_title_font_size)),
                                             range = [par[4] - 0.2, round(par[5] + 1) + 0.2],
                                             domain = xaxis_domain,
                                             gridwidth = 2,
                                             ticklen = 5),
                                yaxis = dict(title = dict(text = par[3],
                                                          font = dict(size = yaxis_title_font_size)),
                                             range = [par[4] - 0.2, round(par[5] + 1) + 0.2],
                                             domain = yaxis_domain,
                                             gridwidth=2,
                                             ticklen=5),
                                legend = dict(x = legend_x,
                                              y = legend_y,
                                              bgcolor = legend_bgcolor),
                                annotations = [dict(x = ann_subtitle_x,
                                                    y = ann_subtitle_y,
                                                    text = par[6],
                                                    font = dict(size = ann_subtitle_font_size),
                                                    showarrow = False,
                                                    xanchor = "left",
                                                    yanchor = "auto",
                                                    xref = 'paper',
                                                    yref = 'paper',
                                                    align = 'left'),
                                               dict(x = ann_text_x,
                                                    y = ann_text_y,
                                                    text = par[7],
                                                    font = dict(size = ann_text_font_size),
                                                    showarrow = False,
                                                    xanchor = "left",
                                                    yanchor = "auto",
                                                    xref = 'paper',
                                                    yref = 'paper',
                                                    align = 'left')])
        
#%%
def generate_box_trace(trace_name, all_box_data, names, colors):
    ct = 1
    for par in zip(trace_name, all_box_data, names, colors):
        if (ct%2 == 1):
            xrefer = ct; yrefer = xrefer + 1
        globals()[par[0]] = go.Box(
                                x = par[1],
                                name = par[2],
                                legendgroup = par[2],
                                boxpoints = 'outliers',
                                marker = dict(color = par[3]),
                                line = dict(color = par[3]),
                                xaxis = 'x{}'.format(xrefer),
                                yaxis = 'y{}'.format(yrefer),
                                showlegend = False
                                )
        ct += 1

#%%
def generate_hist(hist_fig, hist_data):
    for par in zip(hist_fig, hist_data):
        globals()[par[0]] = ff.create_distplot(par[1], 
                                              ['Channel 2', 'Channel 4'],
                                              colors = ['rgba(43, 205, 193, 1)', 'rgba(246, 96, 149, 1)'],
                                              show_hist = False, 
                                              show_curve = True, 
                                              show_rug = False)
        

#%%

def NP_analysis(ch2, ch4, 
                ann_path = 'X:/jeff/Microarray/BanffC88NPAccuracy/annotation_files', 
                quantile_normalization = True,
                ):
    
    ann = pd.read_csv(ann_path + '/Cent_M015_C88_annot.tab')
    ann = ann[ann.probe_id.str.contains('CEN-NP')][['ref', 'x', 'y']]   
    ann.y = 495 - ann.y #Reverse the yaxis
    
    
    
    #Background Annotation File
    x = np.zeros(72*7*7, int)
    y = np.zeros(72*7*7, int)
    
    ct = 0
    for i in range(496): #Y
        for j in range(496): #X
            if((i%81) in range(10)):
                if((j%81) in range(10)):
                    if((i%81) in (1, 8)):
                        if(j%81 in (0, 9)):
                            x[ct] = j
                            y[ct] = i
                            ct += 1
                            continue
                        else:
                            continue
                    if((i%81) in (2, 3, 4, 5, 6, 7)):
                        if((j%81) in (1, 8)):
                            continue                    
                    x[ct] = j
                    y[ct] = i
                    ct += 1
    
    bg_ann = pd.DataFrame()
    bg_ann['x'] = x
    bg_ann['y'] = y
    bg_ann['ref'] = 'bg'
    
    raw = np.stack((ch2['mean'].values, ch4['mean'].values), axis = 1)
    tmp = quantile_norm(raw)
    
    #Processed Data
    ch2_raw = pd.DataFrame()
    ch4_raw = pd.DataFrame()
    ch2_normalized = pd.DataFrame()
    ch4_normalized = pd.DataFrame()
    
    ch2_raw[['x', 'y']] = ch2[['x', 'y']]
    ch4_raw[['x', 'y']] = ch4[['x', 'y']]
    ch2_normalized[['x', 'y']] = ch2[['x', 'y']]
    ch4_normalized[['x', 'y']] = ch4[['x', 'y']]
    
    ch2_raw['mean'] = raw[:,0]
    ch4_raw['mean'] = raw[:,1]
    
    ch2_normalized['mean'] = tmp[:,0]
    ch4_normalized['mean'] = tmp[:,1]
    
    del ch2
    del ch4
    del raw
    del tmp
    
    #Merge Processed Data and Annotation Files
    tp1_raw = pd.merge(ch2_raw, ch4_raw, left_on = ['x', 'y'], right_on = ['x', 'y'], suffixes = ('_2', '_4'))
    res_raw = pd.merge(tp1_raw, ann, left_on = ['x', 'y'], right_on = ['x', 'y'])
    res_raw.loc[(res_raw.ref == 'C') | (res_raw.ref == 'G'), 'label'] = 2
    res_raw.loc[(res_raw.ref == 'A') | (res_raw.ref == 'T'), 'label'] = 4
    All_bg_raw = pd.merge(tp1_raw, bg_ann, left_on = ['x', 'y'], right_on = ['x', 'y'])
    del tp1_raw
    
    tp1_normalized = pd.merge(ch2_normalized, ch4_normalized, left_on = ['x', 'y'], right_on = ['x', 'y'], suffixes = ('_2', '_4'))
    res_normalized = pd.merge(tp1_normalized, ann, left_on = ['x', 'y'], right_on = ['x', 'y'])
    res_normalized.loc[(res_normalized.ref == 'C') | (res_normalized.ref == 'G'), 'label'] = 2
    res_normalized.loc[(res_normalized.ref == 'A') | (res_normalized.ref == 'T'), 'label'] = 4
    All_bg_normalized = pd.merge(tp1_normalized, bg_ann, left_on = ['x', 'y'], right_on = ['x', 'y'])
    del tp1_normalized

    ## scatter data preparation #################################################################################
    # scatter plot argument preparation (raw)
    # stage I (Data dependent aerguments)
    label_2 = res_raw[res_raw.label == 2]
    label_4 = res_raw[res_raw.label == 4]
    critical, ann_text, pred_corr_label_raw = generate_LDA_ann_text(res = res_raw, All_bg = All_bg_raw, datatype = "Raw Data")
    data_x = [np.log2(label_2.mean_2), [0], np.log2(label_4.mean_2), [0], 
              [0, critical[0], 16], [critical[0], critical[0]]]
    data_y = [np.log2(label_2.mean_4), [0], np.log2(label_4.mean_4), [0], 
              [critical[1], critical[1], critical[1]], [0, 16]]
    minimum = np.amin(np.log2(label_2.loc[:, ['mean_2', 'mean_4']]).values)
    maximum = np.amax(np.log2(label_2.loc[:, ['mean_2', 'mean_4']]).values)
    
    # stage II
    colors = ['rgba(43, 205, 193, 1)', 'rgba(246, 96, 149, 1)']
    trace_name = ['trace1', 'trace1_1', 'trace2', 'trace2_1', 'trace5_1', 'trace5_2']
    modes = ['markers', 'markers', 'markers', 'markers', 'lines+markers', 'lines+markers']
    names = ['Label: 2', 'Label: 2', 'Label: 4', 'Label: 4', '3·SD', '3·SD']
    markers = [dict(size = 2, color = colors[0].replace('1)', '0.8)')), 
               dict(size = 6, color = colors[0].replace('1)', '0.8)')),
               dict(size = 2, color = colors[1].replace('1)', '0.8)')),
               dict(size = 6, color = colors[1].replace('1)', '0.8)')),
               dict(size = 10, color = 'rgba(31, 119, 180, 1)'.replace('1)', '0.8)'),
                    line = dict(width = 2, color = 'rgba(31, 119, 180, 1)')),
               dict(size = 10, color = 'rgba(31, 119, 180, 1)'.replace('1)', '0.8)'),
                    line = dict(width = 2, color = 'rgba(31, 119, 180, 1)'))]
    lines =[dict(), dict(), dict(), dict(), 
            dict(color = 'rgba(31, 119, 180, 1)', width = 1.5, dash = 'dash'),
            dict(color = 'rgba(31, 119, 180, 1)', width = 1.5, dash = 'dash')]
    legendgroups = ['group1', 'group1', 'group2', 'group2', "group5", "group5"]
    showlegends = [False, True, False, True, False, True]
    
    #stageIII (Layout arguments)
    scatter_titles = ['<b>Scatter Plot for Raw NP Probes Data (Log Channel 2 v.s. Log Channel 4)</b>']*2
    layout_name = ['layout_html', 'layout_pdf']
    x_titles = ['Log2 Channel 2 Intensity']*2
    y_titles = ['Log2 Channel 4 Intensity']*2
    ann_subtitle = ['<b>Linear Discriminant Analysis Results</b>']*2
    file_formats = ['html', 'pdf']
    
    # generate scatter trace (raw)
    generate_scatter_trace(trace_name = trace_name, all_scatter_data_x = data_x, 
                           all_scatter_data_y = data_y, modes = modes, names = names,
                           markers = markers, lines = lines, legendgroups = legendgroups,
                           showlegends = showlegends)
    # plot HTML
    # generate scatter HTML layout (raw)
    generate_scatter_layout(layout_name = layout_name, scatter_titles = scatter_titles, 
                            x_titles = x_titles, y_titles = y_titles, 
                            data_min = [minimum]*2, data_max = [maximum]*2, 
                            ann_subtitle = ann_subtitle, ann_text = ann_text*2, 
                            file_formats = file_formats)
    data = [trace1, trace1_1, trace2, trace2_1, trace5_1, trace5_2]
    fig = go.Figure(data = data, layout = layout_html)
    plotly.offline.plot(fig, 
            filename = 'Scatter_plots_raw.html', 
            auto_open = False)
    
    # plot pdf
    # generate scatter pdf layout (raw)
    data = [trace1, trace1_1, trace2, trace2_1, trace5_1, trace5_2]
    fig = go.Figure(data = data, layout = layout_pdf)
    pio.write_image(fig, 'Scatter_plots_raw.pdf', 
                    width = 297*3, height = 210*3, scale = 2)

    # scatter plot argument preparation (normalized)
    # stage I (Data dependent variables)
    label_2 = res_normalized[res_normalized.label == 2]
    label_4 = res_normalized[res_normalized.label == 4]
    critical, ann_text, pred_corr_label_normalized = generate_LDA_ann_text(res = res_normalized, All_bg = All_bg_normalized, datatype = "Normalized Data")
    
    data_x = [np.log2(label_2.mean_2), [0], np.log2(label_4.mean_2), [0], 
              [0, critical[0], 16], [critical[0], critical[0]]]
    data_y = [np.log2(label_2.mean_4), [0], np.log2(label_4.mean_4), [0], 
              [critical[1], critical[1], critical[1]], [0, 16]]
    minimum = np.amin(np.log2(label_2.loc[:, ['mean_2', 'mean_4']]).values)
    maximum = np.amax(np.log2(label_2.loc[:, ['mean_2', 'mean_4']]).values)
    
    #stageIII (Layout arguments)
    scatter_titles = ['<b>Scatter Plot for Normalized NP Probes Data (Log Channel 2 v.s. Log Channel 4)</b>']*2
    
    # generate scatter trace (normalized)
    generate_scatter_trace(trace_name = trace_name, all_scatter_data_x = data_x, 
                           all_scatter_data_y = data_y, modes = modes, names = names,
                           markers = markers, lines = lines, legendgroups = legendgroups,
                           showlegends = showlegends)
    # plot HTML
    # generate scatter HTML layout (normalized)
    generate_scatter_layout(layout_name = layout_name, scatter_titles = scatter_titles, 
                            x_titles = x_titles, y_titles = y_titles, 
                            data_min = [minimum]*2, data_max = [maximum]*2, 
                            ann_subtitle = ann_subtitle, ann_text = ann_text*2, 
                            file_formats = file_formats)
    data = [trace1, trace1_1, trace2, trace2_1, trace5_1, trace5_2]
    fig = go.Figure(data = data, layout = layout_html)
    plotly.offline.plot(fig, 
            filename = 'Scatter_plots_normalized.html', 
            auto_open = False)
    
    # plot pdf
    # generate scatter pdf layout (normalized)
    data = [trace1, trace1_1, trace2, trace2_1, trace5_1, trace5_2]
    fig = go.Figure(data = data, layout = layout_pdf)
    pio.write_image(fig, 'Scatter_plots_normalized.pdf', 
                    width = 297*3, height = 210*3, scale = 2)
    
    ## Box Plot and Density plot ###################################################
    # Add histogram data
    ch2_log_bg_raw = np.log2(All_bg_raw.mean_2.values) #background ch2 data
    ch4_log_bg_raw = np.log2(All_bg_raw.mean_4.values) #background ch4 data
    
    ch2_log_np_raw = np.log2(res_raw.mean_2.values)
    ch4_log_np_raw = np.log2(res_raw.mean_4.values)
    
    ch2_log_bg_normalized = np.log2(All_bg_normalized.mean_2.values) #background ch2 data
    ch4_log_bg_normalized = np.log2(All_bg_normalized.mean_4.values) #background ch4 data
    
    ch2_log_np_normalized = np.log2(res_normalized.mean_2.values)
    ch4_log_np_normalized = np.log2(res_normalized.mean_4.values)
    
    trace_name = ['trace1', 'trace2', 'trace3', 'trace4', 'trace5', 'trace6', 'trace7', 'trace8']
    all_box_data = [ch2_log_bg_raw, ch4_log_bg_raw, ch2_log_np_raw, ch4_log_np_raw, 
                    ch2_log_bg_normalized, ch4_log_bg_normalized, ch2_log_np_normalized, ch4_log_np_normalized]
    names = ['Channel 2', 'Channel 4']*4
    colors = ['rgba(43, 205, 193, 1)', 'rgba(246, 96, 149, 1)']*4
    hist_fig = ['fig_bg_raw', 'fig_np_raw', 'fig_bg_normalized', 'fig_np_normalized']
    hist_data = [[ch2_log_bg_raw, ch4_log_bg_raw], [ch2_log_np_raw, ch4_log_np_raw],
                 [ch2_log_bg_normalized, ch4_log_bg_normalized], [ch2_log_np_normalized, ch4_log_np_normalized]]
    
    # plot box plot
    generate_box_trace(trace_name, all_box_data, names, colors = colors)
    # plot histgoram
    generate_hist(hist_fig, hist_data)

    # Combine boxplot and histogram
    All_hist_fig = [fig_bg_raw, fig_np_raw, fig_bg_normalized, fig_np_normalized]
    All_trace = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8]
    
    for i in range(len(All_hist_fig)):
        fig = All_hist_fig[i]
        for j in range(len(fig.data)):
            fig.data[j].update(xaxis = 'x{}'.format(2*i + 1))
            fig.data[j].update(yaxis = 'y{}'.format(2*i + 1))
            fig.data[j].update(showlegend = False)
            fig.add_traces([All_trace[2*i + j]])
    
    
    
    # Raw Background
    fig_bg_raw['layout'] = {}
    fig_bg_raw.layout.update(
        title = "",
        xaxis1 = dict(
                anchor = 'y2',    #axis apperance
                title = '',
                domain = [0, 0.45],
                range = [5, 15]
                ),
        yaxis1 = dict(
                anchor = 'x1',
                title = dict(text = 'Probability Density', 
                             font = dict(size = 8)),
                domain = [0.7, 1]
                ),
        xaxis2 = dict(
                anchor = 'y2',    #axis apperance
                title = '',
                domain = [0, 0.45],
                range = [5, 15]
                ),
        yaxis2 = dict(
                anchor = 'x2',
                domain = [0.55, 0.68]
                ),
        showlegend = False
        )
    
    # Raw NP
    fig_np_raw['layout'] = {}
    fig_np_raw.layout.update(
        title = "",
        xaxis3 = dict(
                anchor = 'y4', 
                title = '',
                domain = [0.55, 1],
                range = [5, 15]
                ),
        yaxis3 = dict(
                anchor = 'x3',
                title = '',
                domain = [0.7, 1]
                ),
        xaxis4 = dict(
                anchor = 'y4', 
                title = '',
                domain = [0.55, 1],
                range = [5, 15]
                ),
        yaxis4 = dict(
                anchor = 'x4',
                domain = [0.55, 0.68]
                ),
        showlegend = False
        )
    
    # Normalized background
    fig_bg_normalized['layout'] = {}
    fig_bg_normalized.layout.update(
        title = "",
        xaxis5 = dict(
                anchor = 'y6',    #axis apperance
                title = dict(text = 'Log2 Intensity',
                             font = dict(size = 8)),
                domain = [0, 0.45],
                range = [5, 15]
                ),
        yaxis5 = dict(
                anchor = 'x5',
                title = dict(text = 'Probability Density', 
                             font = dict(size = 8)),
                domain = [0.15, 0.45]
                ),
        xaxis6 = dict(
                anchor = 'y6',    #axis apperance
                title = dict(text = 'Log2 Intensity',
                             font = dict(size = 8)),
                domain = [0, 0.45],
                range = [5, 15]
                ),
        yaxis6 = dict(
                anchor = 'x6',
                domain = [0, 0.13]
                ),
        showlegend = False
        )
    
    # Normalized NP
    fig_np_normalized['layout'] = {}
    fig_np_normalized.layout.update(
        title = dict(text = "<b>Distribution and boxplot for background and NP probes</b>",
                     font = dict(size = 12)),
        xaxis7 = dict(
                anchor = 'y8', 
                title = dict(text = 'Log2 Intensity',
                             font = dict(size = 8)),
                domain = [0.55, 1],
                range = [5, 15]
                ),
        yaxis7 = dict(
                anchor = 'x7',
                title = '',
                domain = [0.15, 0.45]
                ),
        xaxis8 = dict(
                anchor = 'y8', 
                title = dict(text = 'Log2 Intensity',
                             font = dict(size = 8)),
                domain = [0.55, 1],
                range = [5, 15]
                ),
        yaxis8 = dict(
                anchor = 'x8',
                domain = [0, 0.13]
                ),
        showlegend = False,
        annotations = [
                dict(x = 0.225,
                     y = 1.03,
                     showarrow = False,
                     text = 'Raw Background Probes Data',
                     font = dict(size = 10),
                     xanchor = 'center',
                     yanchor = 'bottom',
                     xref = 'paper',
                     yref = 'paper'
                        ),
                dict(x = 0.225,
                     y = 0.48,
                     showarrow = False,
                     text = 'Normalized Background Probes Data',
                     font = dict(size = 10),
                     xanchor = 'center',
                     yanchor = 'bottom',
                     xref = 'paper',
                     yref = 'paper'
                        ),
                dict(x = 0.775,
                     y = 1.03,
                     showarrow = False,
                     text = 'Raw NP Probes Data',
                     font = dict(size = 10),
                     xanchor = 'center',
                     yanchor = 'bottom',
                     xref = 'paper',
                     yref = 'paper'
                        ),
                dict(x = 0.775,
                     y = 0.48,
                     showarrow = False,
                     text = 'Normalized NP Probes Data',
                     font = dict(size = 10),
                     xanchor = 'center',
                     yanchor = 'bottom',
                     xref = 'paper',
                     yref = 'paper'
                        )                 
                ]
        )
    
    ##################################################################################
    fig = go.Figure()
    fig.add_traces([fig_bg_raw.data[0],
                    fig_bg_raw.data[1],
                    fig_bg_raw.data[2],
                    fig_bg_raw.data[3],
                    fig_np_raw.data[0],
                    fig_np_raw.data[1],
                    fig_np_raw.data[2],
                    fig_np_raw.data[3],
                    fig_bg_normalized.data[0],
                    fig_bg_normalized.data[1],
                    fig_bg_normalized.data[2],
                    fig_bg_normalized.data[3],
                    fig_np_normalized.data[0],
                    fig_np_normalized.data[1],
                    fig_np_normalized.data[2],
                    fig_np_normalized.data[3]])
    
    
    fig.layout.update(fig_bg_raw.layout)
    fig.layout.update(fig_np_raw.layout)
    fig.layout.update(fig_bg_normalized.layout)
    fig.layout.update(fig_np_normalized.layout)
    
    # Plot!
    pio.write_image(fig = fig, file = 'Distribution_plots.pdf',
                    width = 297*3, height = 210*3, scale = 2)
    plotly.offline.plot(fig, 
            filename = 'Distribution_plots.html', 
            auto_open = False)
    
    #%% Heatmap (Only for normalized data)
    ch2_matri_normalized = np.zeros((496, 496))
    ch4_matri_normalized = np.zeros((496, 496))
    
    for x, y, m2, m4 in res_normalized.loc[:, ['x', 'y', 'mean_2', 'mean_4']].values:
        ch2_matri_normalized[int(y)][int(x)] = np.log2(m2)
        ch4_matri_normalized[int(y)][int(x)] = np.log2(m4)
    
    panel = ch2_matri_normalized + ch4_matri_normalized
    nnzeros = panel.sum(axis=1)
    nnzeros = np.convolve(nnzeros, [1, 1, 1, 1, 1], mode = 'same')
    nnzeros = np.where(nnzeros)[0]
    
    compare_NP_heat = go.Heatmap(z = (ch4_matri_normalized[nnzeros,:] - ch2_matri_normalized[nnzeros,:]),
                                 zmin = -1, zmax = 1,
                                 colorscale=[[0, 'rgba(0, 255 ,0, 1)'], 
                                             [0.5, 'rgba(255, 255 ,0, 1)'] , 
                                             [1, 'rgba(255, 0, 0, 1)']])
    
    data = [compare_NP_heat]
    layout = go.Layout(title = '<b>Heatmap for Normalized NP Probes (log(CH4) - log(CH2))</b>', 
                       width = 1200,
                       height = 500,
                       xaxis = dict(side = "top"),
                       yaxis = dict(autorange = "reversed")
                       )
    fig = go.Figure(data = data, layout = layout)
    
    plotly.offline.plot(fig, 
            filename = 'Heatmap_normalized_NP_compare.html', 
            auto_open = False)
    
    pio.write_image(fig = fig, file = 'Heatmap_normalized_NP_compare.pdf', 
                    width = 297*3, height = 210*3, scale = 2)
    
    
    #%% misclassified normalized NP Probes
    bg = np.zeros((496, 496))
    
    for x, y in res_normalized.loc[~pred_corr_label_normalized, ['x', 'y']].values:
        bg[int(y)][int(x)] = 1
    
    compare_NP_heat = go.Heatmap(z = bg[nnzeros,:],
                                 zmin = 0, zmax = 1,
                                 colorscale=[[0, 'rgba(0, 0 ,0, 1)'], 
                                              [1, 'rgba(255, 255, 255, 1)']])
    
    data = [compare_NP_heat]
    layout = go.Layout(title = '<b>Heatmap for Misclassified NP Probes by LDA (Normalized)</b>', 
                       width = 1200,
                       height = 500,
                       xaxis = dict(side = "top", domain = [0, 0.987]),
                       yaxis = dict(autorange = "reversed")
                       )
    fig = go.Figure(data = data, layout = layout)
    
    plotly.offline.plot(fig, 
            filename = 'Heatmap_misclassified_normalized.html', 
            auto_open = False)
    
    pio.write_image(fig = fig, file = 'Heatmap_misclassified_normalized.pdf', 
                    width = 297*3, height = 210*3, scale = 2)
    
#%%
    




## Main Program #################################################################
if(platform.system() == 'Linux'):
    desktop_wd = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
else:
    desktop_wd = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

def make_directories(wd, input_list):
    flag = []
    for string in input_list:
        dirpath = os.path.join(wd, string)
        try:
            os.mkdir(dirpath)
        except FileExistsError:
            flag.append(0)
            continue
        else:
            flag.append(1)
            print('Directory {} created'.format(dirpath))
    return flag

folder_name = 'QC_analysis_results'
make_directories(desktop_wd, [folder_name])
QCwd = desktop_wd + "\\" + folder_name
os.chdir(QCwd)

filepath = 'X:\\john\\DVT3_rotate_fail'
update = make_directories(QCwd, os.listdir(filepath))

old = ['X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\324_20190416142802',
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\339_20190416143443',
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\61_20190416141123',
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\51_20190416140803', 
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\46_20190416140441', 
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\36_20190416140122', 
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\349_20190416143802',
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\339_20190416143443', 
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\334_20190416143122', 
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\324_20190416142802', 
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\205_20190416142443',
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\195_20190416142122', 
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\190_20190416141800', 
       'X:\\john\\DVT3_rotate_fail\\20190416_1NIF880-23#14_DVT3\\180_20190416141440']

new = ['X:\\john\\DVT3_rotate_fail\\wafer878-01\\plate1\\C046_2019_3_19_12_24_23_Pr',
       'X:\\john\\DVT3_rotate_fail\\wafer878-04\\plate17_clean\\C046_2019_3_14_15_13_58_Pr',
       'X:\\john\\DVT3_rotate_fail\\wafer877-01\\plate5\\C036_2019_3_1_11_39_16_Ax',
       'X:\\john\\DVT3_rotate_fail\\wafer878-01\\plate1\\C035_2019_3_19_12_18_32_Ax']


for i in range(len(update)):
    if (i != 0):
        if (update[i]) :
            smt_lay1 = os.path.join(filepath, os.listdir(filepath)[i])
            folder_lay1 = os.path.join(QCwd, os.listdir(filepath)[i])
            for j in range(len(os.listdir(smt_lay1))):
                smt_lay2 = os.path.join(smt_lay1, os.listdir(smt_lay1)[j])
                if(smt_lay2.find('plate') != -1):
                    make_directories(folder_lay1, [os.listdir(smt_lay1)[j]])
                    folder_lay2 = os.path.join(folder_lay1, os.listdir(smt_lay1)[j])
                    for k in range(len(os.listdir(smt_lay2))):
                        smt_lay3 = os.path.join(smt_lay2, os.listdir(smt_lay2)[k])
                        if(smt_lay3 in new):
                            continue  #No Data on smtdata
                        datapath = []
                        for foldName, subfolders, filenames in os.walk(smt_lay3):
                            for filename in filenames:
                                if filename.endswith('heatmap.csv'):
                                    print(foldName)
                                    
                                    temp = foldName.split('\\')
                                    chip_id = temp[5].split('_')[0]
                                    chip_id_date = temp[5]
                                    make_directories(folder_lay2, [chip_id_date])
                                    os.chdir(os.path.join(folder_lay2, chip_id_date))
                                    datapath.append(foldName.replace("\\", '/') + '/' + filename)
                        print("OK")
                        #Load Data
                        target = ["/2/", "/green/", "/CY3/"]
                        if(any(elem in datapath[0] for elem in target)):
                            ch2 = pd.read_csv(datapath[0]).loc[:, ['x', 'y', 'mean']]
                            ch4 = pd.read_csv(datapath[1]).loc[:, ['x', 'y', 'mean']]
                        else:
                            ch2 = pd.read_csv(datapath[1]).loc[:, ['x', 'y', 'mean']]
                            ch4 = pd.read_csv(datapath[0]).loc[:, ['x', 'y', 'mean']]

                        print(ch2.head(3))
                        print(ch4.shape)
                        NP_analysis(ch2, ch4)
                        
                else:
                    if(smt_lay2 in old):
                            continue   #runtime error
                    datapath = []
                    for foldName, subfolders, filenames in os.walk(smt_lay2):
                        for filename in filenames:
                            if filename.endswith('heatmap.csv'):
                                print(foldName)
                                
                                temp = foldName.split('\\')
                                chip_id = temp[4].split('_')[0]
                                chip_id_date = temp[4]
                                make_directories(folder_lay1, [chip_id_date])
                                os.chdir(os.path.join(folder_lay1, chip_id_date))
                                datapath.append(foldName.replace("\\", '/') + '/' + filename)
                    
                    #Load Data
                    target = ["/2/", "/green/", "/CY3/"]
                    if(any(elem in datapath[0] for elem in target)):
                        ch2 = pd.read_csv(datapath[0]).loc[:, ['x', 'y', 'mean']]
                        ch4 = pd.read_csv(datapath[1]).loc[:, ['x', 'y', 'mean']]
                    else:
                        ch2 = pd.read_csv(datapath[1]).loc[:, ['x', 'y', 'mean']]
                        ch4 = pd.read_csv(datapath[0]).loc[:, ['x', 'y', 'mean']]
                    
                    print(ch2.head(3))
                    print(ch4.shape)
                    NP_analysis(ch2, ch4)
                
                

os.chdir(desktop_wd)
