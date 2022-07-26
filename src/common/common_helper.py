# import Python libraries
from tracemalloc import start
import pandas as pd
import numpy as np
from datetime import datetime, date
import datetime as dt
from IPython.core.interactiveshell import InteractiveShell
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from scikitplot.metrics import plot_cumulative_gain
from scikitplot.metrics import plot_lift_curve
from scikitplot.metrics import plot_roc
from scikitplot.metrics import plot_precision_recall
from scikitplot.metrics import plot_ks_statistic
import sklearn
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import io
import os
from typing import Dict

InteractiveShell.ast_node_interactivity = "all"
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format


def condense_category(col: pd.Series, new_name: str='other', min_freq: float=0.01, min_count: int=None):
    if min_count:
        series = pd.value_counts(col)
        mask = (series).lt(min_count)
        return pd.Series(np.where(col.isin(series[mask].index), new_name, col))
    else:
        series = pd.value_counts(col)
        mask = (series/series.sum()).lt(min_freq)
        return pd.Series(np.where(col.isin(series[mask].index), new_name, col))


def encode_and_bind(df: pd.DataFrame, feature_to_encode: str)-> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame        
    feature_to_encode : str
        feature to encode

    Returns
    -------
    pd.DataFrame
        Dataframe, drop original feature and add the encoded feature
    """
    dummies = pd.get_dummies(df[[feature_to_encode]])
    res = pd.concat([df, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 


def get_season(datecol, Y=2017):
    seasons = [('invierno', (date(Y,  6,  21),  date(Y,  9, 20))),
           ('primavera', (date(Y,  9, 21),  date(Y,  12, 20))),
           ('Verano', (date(Y,  12, 21),  date(Y,  12, 31))),
           ('Verano', (date(Y,  1, 1),  date(Y,  3, 20))),
           ('otonio', (date(Y,  3, 21),  date(Y, 6, 20))),]
    if isinstance(datecol, datetime):
        datecol = datecol.date()
    datecol = datecol.replace(year=Y)
    return next(season for season, (start, end) in seasons
                if start <= datecol <= end)


def completeness_variability(df: pd.DataFrame, threshold_var: float = 0.05, threshold_null: float = 0.90,
                             path_output:str = None):
    """It takes a set of features and returns the completeness and variability of each feature. In order to discard features that overcome ``threshold_var`` or ``threshold_null``.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with the features.
    threshold_var : float, optional
        Threshold to remove features with low variability , by default 0.05
    threshold_null : float, optional
        Threshold to remove features with low completeness, by default 0.90
    path_output : str, optional
        path of the output, by default None
    Returns
    -------
    CSV file
        It returns a file with the completeness and variability of all the features. and the regards mark keep or not.
    """
    completeness = pd.DataFrame(df.isnull().sum() / df.shape[0], columns=['Missing_Percentage']).reset_index()
    p_start = pd.DataFrame(df.quantile(threshold_var).reset_index())
    p_end = pd.DataFrame(df.quantile(1 - threshold_var).reset_index())
    variability = p_start.merge(p_end, how='left', on='index')
    start = str(threshold_var * 100)
    end = str((1 - threshold_var) * 100)
    variability['Percentil_' + start] = variability.iloc[:, 1]
    variability['Percentil_' + end] = variability.iloc[:, 2]
    variability['Variability'] = np.where(variability['Percentil_' + start] == variability['Percentil_' + end],
                                          'Not variable',
                                          np.where(((variability['Percentil_' + start]).isnull()) & (
                                              (variability['Percentil_' + end]).isnull()),
                                                   'No variable', 'Variable'))
    variability = variability[['index', 'Percentil_' + start, 'Percentil_' + end, 'Variability']]
    cv = completeness.merge(variability, how='left', on='index')
    cv['completeness'] = np.where(cv['Missing_Percentage'] > threshold_null,
                                  'Missing > ' + str(threshold_null*100) + '%',
                                  'Missing <= ' + str(threshold_null*100) + '%')
    cv['final_decision'] = np.where((cv['Variability'] == 'Not variable') |
                                    (cv['Missing_Percentage'] > threshold_null), 'Exclude', 'Include')
    keepcolumns = list(cv[cv['final_decision'] == 'Include']['index'].values)

    if (path_output is not None) :
        cv.to_csv(path_output, index=False, sep='|')

    return {'cv': cv,
            'keepcolumns': keepcolumns}


def validation_table(
                        df: pd.DataFrame,
                        target: str ,
                        prob: str ,
                        bins: int = 10,
                        custom_bins: list= None,
                        path_output: str = None,
                        type_output: str = 'console',
                        plots: bool = True,
                        ):
    """This function calculates a validation table and the regardless graphs to check the accuracy and performance of a classification model.
    It shows KS, GINI, AUC, AR, Lift, Precision, Recall, and ROC curves.
    Args:
        df (pd.DataFrame): Pandas dataframe with the target variable and the probability
        target (str): name of the target variable in df
        prob (str): name of the probability/propensity variable in df
        bins (int, optional): Number of intervals to generate in validation table. Defaults to 10.
        custom_bins (list, optional): Personalized intervals for validation table. Defaults to None.
        path_output (str, optional): Path to save the html/xlsx output. Defaults to None.
        type_output (str, optional): Output type, "console", "xlsx" or "html". Defaults to None.
    Returns:
        None: It returns a valitation table and the performance Graphs for a binary classification model. Depending on the type of output, it can be saved in a html file, a xlsx file or in the console.
    """
    scorecard = {}
    dfsc = df.copy()
    auc = roc_auc_score(dfsc[target], dfsc[prob])
    dfsc['Objetivo'] = 1 - dfsc[target]
    if custom_bins is None:
        dfsc['bucket'] = pd.qcut(dfsc[prob], bins)
    else:
        dfsc['bucket'] = pd.cut(dfsc[prob], bins=custom_bins, include_lowest=True)
        bins = len(custom_bins)-1
    grouped = dfsc.groupby('bucket', as_index=False)
    kstable = pd.DataFrame()
    kstable['Min_prob'] = grouped.min()[prob]
    kstable['Max_prob'] = grouped.max()[prob]
    kstable['Total'] = grouped.count()[prob]
    kstable['Event'] = grouped.sum()[target]
    kstable['NoEvent'] = grouped.sum()['Objetivo']
    kstable = kstable.sort_values(by="Min_prob", ascending=True).reset_index(drop=True)
    kstable['Event_Rate'] = (kstable.Event / kstable.Total)
    kstable['NoEvent_Rate'] = (kstable.NoEvent / kstable.Total)
    kstable['RateAcum_EventUP'] = (kstable.Event.cumsum() / kstable.Event.sum())
    kstable.sort_index(ascending=False, inplace=True)
    kstable['RateAcum_EventDOWN'] = (kstable.Event.cumsum() / kstable.Event.sum())
    kstable.sort_index(ascending=True, inplace=True)
    kstable['RateAcum_NoEvent'] = (kstable.NoEvent.cumsum() / kstable.NoEvent.sum())
    kstable['KS'] = (abs(kstable['RateAcum_EventUP'] - kstable['RateAcum_NoEvent']) * 100)
    kstable['RateUP'] = (kstable['Event'].cumsum() / kstable['Total'].cumsum())
    kstable.sort_index(ascending=False, inplace=True)
    kstable['RateDOWN'] = (kstable['Event'].cumsum() / kstable['Total'].cumsum())
    kstable.sort_index(ascending=True, inplace=True)
    kstable['Odds'] = (kstable['NoEvent'] / kstable['Event'])
    kstable['Odds_Acum'] = (kstable['NoEvent'].cumsum() / kstable['Event'].cumsum())
    kstable['Lift'] = ((kstable['Event_Rate']) /
                       ((kstable['Event'].sum() / kstable['Total'].sum()) / (
                               kstable['NoEvent'].sum() / kstable['Total'].sum())))
    kstable['breaks'] = np.where(kstable['Event_Rate'].shift(1) > kstable['Event_Rate'], 1, 0)
    kstable.index = range(1, bins + 1)
    kstable.index.names = ['Decil']
    resumen = pd.DataFrame(columns=['Population', 'Events', 'Event_Rate', 'KS', 'AUC',
                                    'AR', 'Gini', 'Odds_sup', 'Odds_inf', 'Breaks'])
    resumen.at[0, 'Population'] = kstable.Total.sum()
    resumen.at[0, 'Events'] = kstable.Event.sum()
    resumen.at[0, 'Event_Rate'] = resumen['Events'][0] / resumen['Population'][0]
    resumen.at[0, 'KS'] = kstable.KS.max()
    resumen.at[0, 'AR'] = 2 * auc - 1
    resumen.at[0, 'Gini'] = (2 * auc - 1) * (1 - resumen['Events'][0] / resumen['Population'][0])
    resumen.at[0, 'AUC'] = auc
    resumen.at[0, 'Odds_sup'] = kstable.Odds.max()
    resumen.at[0, 'Odds_inf'] = kstable.Odds.min()
    resumen.at[0, 'Breaks'] = kstable.breaks.sum()
    kstable.drop(columns=['RateAcum_NoEvent', 'breaks'], inplace=True)
    scorecard['KS_total'] = kstable
    scorecard['resumen'] = resumen    
    pd.set_option('display.width', 400)
    pd.set_option('display.max_columns', None)
    ks_print = kstable.copy()
    ks_print["Min_prob"] = ks_print["Min_prob"].apply("{0:.2f}".format)
    ks_print["Max_prob"] = ks_print["Max_prob"].apply("{0:.2f}".format)
    ks_print["Total"] = ks_print["Total"].apply("{:,}".format)
    ks_print["Event"] = ks_print["Event"].apply("{:,}".format)
    ks_print["NoEvent"] = ks_print["NoEvent"].apply("{:,}".format)
    ks_print["RateUP"] = ks_print["RateUP"].apply("{0:.2%}".format)
    ks_print["RateDOWN"] = ks_print["RateDOWN"].apply("{0:.2%}".format)
    ks_print["KS"] = ks_print["KS"].apply("{0:.2f}".format)
    ks_print["Lift"] = ks_print["Lift"].apply("{0:.2f}".format)
    ks_print["Event_Rate"] = ks_print["Event_Rate"].apply("{0:.2%}".format)
    ks_print["NoEvent_Rate"] = ks_print["NoEvent_Rate"].apply("{0:.2%}".format)
    ks_print["RateAcum_EventUP"] = ks_print["RateAcum_EventUP"].apply("{0:.2%}".format)
    ks_print["RateAcum_EventDOWN"] = ks_print["RateAcum_EventDOWN"].apply("{0:.2%}".format)
    ks_print["Odds"] = ks_print["Odds"].apply("{0:.1f}".format)
    ks_print["Odds_Acum"] = ks_print["Odds_Acum"].apply("{0:.1f}".format)
    resumen_print = resumen.copy()
    resumen_print['Population'] = resumen_print['Population'].apply("{:,}".format)
    resumen_print['Events'] = resumen_print['Events'].apply("{:,}".format)
    resumen_print['Event_Rate'] = resumen_print['Event_Rate'].apply("{0:.2%}".format)
    resumen_print['KS'] = resumen_print['KS'].apply("{0:.1f}".format)
    resumen_print['AR'] = (resumen_print['AR'] * 100).apply("{0:.1f}".format)
    resumen_print['Gini'] = (resumen_print['Gini'] * 100).apply("{0:.1f}".format)
    resumen_print['AUC'] = (resumen_print['AUC'] * 100).apply("{0:.1f}".format)
    resumen_print['Odds_sup'] = resumen_print['Odds_sup'].apply("{0:.1f}".format)
    resumen_print['Odds_inf'] = resumen_print['Odds_inf'].apply("{0:.1f}".format)
    resumen_print['Breaks'] = resumen_print['Breaks'].apply("{0:.1f}".format)
    
    if type_output == 'xlsx':
        # graphs
        y_pred = 1 - df[[prob]]
        y_pred['prob_0'] = 1 - y_pred[prob]
        y_pred = y_pred.to_numpy()
        plt.style.use("seaborn")
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(21, 4))
        plot_roc(df[target], y_pred, ax=ax1, plot_micro=False, plot_macro=False, classes_to_plot=1,
                text_fontsize='small')
        plot_cumulative_gain(df[target], y_pred, ax=ax2, text_fontsize='small')
        plot_ks_statistic(df[target], y_pred, ax=ax3, text_fontsize='small')
        plot_lift_curve(df[target], y_pred, ax=ax4, text_fontsize='small')
        plot_precision_recall(df[target], y_pred, ax=ax5, plot_micro=False, classes_to_plot=1, text_fontsize='small')
        ax3.legend(loc='upper left', fontsize='small')
        ax4.legend(loc='upper right', fontsize='small')
        plt.subplots_adjust(wspace=0.2)
        plt.tight_layout()
        with pd.ExcelWriter(path_output+'.xlsx', engine='xlsxwriter') as writer:
            scorecard['resumen'].to_excel(writer, sheet_name='Scorecard', startrow=1, header=False, index=False,
                                            float_format="%.4f")
            scorecard['KS_total'].to_excel(writer, sheet_name='Scorecard', startrow=4, header=False, index=False,
                                            float_format="%.4f")
            workbook = writer.book
            worksheet = writer.sheets['Scorecard']
            cells_format = workbook.add_format({
                'valign': 'vcenter',
                'align': 'center'
            })
            worksheet.set_column('A:A', width=12, cell_format=cells_format)
            worksheet.set_column('B:E', width=10, cell_format=cells_format)
            worksheet.set_column('F:G', width=15, cell_format=cells_format)
            worksheet.set_column('H:I', width=21, cell_format=cells_format)
            worksheet.set_column('J:O', width=10, cell_format=cells_format)
            per_format = workbook.add_format(
                {'num_format': '0.00%', 'bg_color': '#FFFFFF', 'bottom': 1, 'top': 1})
            num_format = workbook.add_format(
                {'num_format': '#,##0', 'bg_color': '#FFFFFF', 'bottom': 1, 'top': 1})
            flo_format = workbook.add_format(
                {'num_format': '0.00', 'bg_color': '#FFFFFF', 'bottom': 1, 'top': 1})
            flo2_format = workbook.add_format(
                {'num_format': '0.0000', 'bg_color': '#FFFFFF', 'bottom': 1, 'top': 1})
            rango = 4 + bins
            worksheet.conditional_format('A5:B' + str(rango), {'type': 'no_errors', 'format': flo2_format})
            worksheet.conditional_format('C5:E' + str(rango), {'type': 'no_errors', 'format': num_format})
            worksheet.conditional_format('F5:F' + str(rango), {'type': '3_color_scale', 'format': per_format})
            worksheet.conditional_format('F5:I' + str(rango), {'type': 'no_errors', 'format': per_format})
            worksheet.conditional_format('J5:J' + str(rango), {'type': 'no_errors', 'format': flo_format})
            worksheet.conditional_format('K5:L' + str(rango), {'type': 'no_errors', 'format': per_format})
            worksheet.conditional_format('M5:O' + str(rango), {'type': 'no_errors', 'format': flo_format})
            worksheet.conditional_format('O5:O' + str(rango), {'type': 'data_bar', 'format': flo_format})
            worksheet.conditional_format('A2:B2', {'type': 'no_errors', 'format': num_format})
            worksheet.conditional_format('C2:C2', {'type': 'no_errors', 'format': per_format})
            worksheet.conditional_format('D2:J2', {'type': 'no_errors', 'format': flo_format})
            header_format = workbook.add_format({
                # 'text_wrap': True,
                'bold': True,
                'valign': 'vcenter',
                'align': 'center',
                'fg_color': '#432E8B',
                'color': '#FFFFFF',
                'border': 1
            })
            for col_num, value in enumerate(scorecard['KS_total'].columns.values):
                worksheet.write(3, col_num, value, header_format)
            for col_num, value in enumerate(scorecard['resumen'].columns.values):
                worksheet.write(0, col_num, value, header_format)
            imgdata = io.BytesIO()
            fig.savefig(imgdata, format='png')
            worksheet.insert_image(rango + 2, 0, '', {'image_data': imgdata})
            # writer.save()
            fig.clear()
            plt.close(fig)
        print('Backtesting in', path_output)
    elif type_output == 'html':
        y_pred = 1 - df[[prob]]
        y_pred['prob_0'] = 1 - y_pred[prob]
        y_pred = y_pred.to_numpy()
        plt.style.use("seaborn")
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        plot_roc(df[target], y_pred, ax=axes[0], plot_micro=False, plot_macro=False, classes_to_plot=1, text_fontsize='small')
        plot_ks_statistic(df[target], y_pred, ax=axes[1], text_fontsize='small')
        plot_lift_curve(df[target], y_pred, ax=axes[2], text_fontsize='small')
        plot_precision_recall(df[target], y_pred, ax=axes[3], plot_micro=False, classes_to_plot=1, text_fontsize='small')
        axes[0].legend(loc='lower right', fontsize='small')
        axes[1].legend(loc='upper left', fontsize='small')
        axes[2].legend(loc='upper right', fontsize='small')
        axes[3].legend(loc='upper right', fontsize='small')
        plt.subplots_adjust(wspace=0.3)
        plt.tight_layout()
        name_plot = path_output[0:path_output.rfind('/')]+'/graph'+path_output[path_output.rfind('/')+1:]
        plt.savefig(name_plot)
        html_string = '''
                        <html>
                        <head><title>HTML Pandas Dataframe with CSS</title>
                            <style>
                                body{{
                                    font-family: Arial;
                                    /* display: flex; */
                                    align-items: center;
                                    justify-content: center;
                                    flex-direction: column;
                                    background-color: #b1b0b0;
                                }}
                                .mystyle {{
                                    font-size: 11pt; 
                                    font-family: Arial;
                                    border-collapse: collapse; 
                                    border: 1px solid rgb(230, 221, 221);
                                }}
                                .mystyle td, th {{
                                    padding: 4px;
                                    text-align: center;
                                }}
                                .mystyle tr:nth-child(even) {{
                                    background:  #dfdcdcd2;
                                }}
                                .mystyle tr:hover {{
                                    background: rgb(110, 84, 145);
                                    cursor: pointer;
                                    color: white;
                                }}
                                .mystyle th {{
                                    background-color: #380d5f;
                                    color: white;
                                }}
                                .container {{
                                    padding: 5px 20px;
                                    /* border: 10px solid red; */
                                    /* justify-content: center; */
                                    box-sizing: border-box;
                                    align-content: center;
                                    display: flex;
                                    flex-direction: column;
                                    background-color: rgb(255, 255, 255)
                                    /* padding-top: 70px; */
                                    /* background: inherit; */
                                    /* position: absolute; */
                                    
                                }}
                                .sep {{
                                    /* border: 1px solid red; */
                                    background-color: #010350;
                                    padding: 5px;
                                    text-align: center;
                                    color: white;
                                    margin: 0%;
                                    padding: 5px;
                                    border-radius: 3px;
                                    font-size: 18pt;
                                    /* background-image: url("../images/logo_c.png"); */
                                    /* opacity: 0.1; */
                                    /* background-repeat: no-repeat; */
                                }}
                                .sep1 {{
                                    /* border: 1px solid red; */
                                    background-color:  #380d5f;
                                    padding: 5px;
                                    text-align: center;
                                    color: rgb(255, 255, 255);
                                    margin: 0%;
                                    padding: 5px;
                                    border: #ffffff solid 1px;
                                    border-radius: 5px;
                                    font-size: 14pt;
                                    /* background-image: url("../images/logo_c.png"); */
                                    /* opacity: 0.1; */
                                    /* background-repeat: no-repeat; */
                                }}
                                img {{
                                    max-width: 100%;
                                    height: auto;
                                    width: auto\9; /* ie8 */
                                    /* border: #E0E0E0 solid 1px; */
                                    border-radius: 6px;
                                    /* padding: 1%; */
                                }}
                                h2, h3 {{
                                    margin: 5px;
                                }}
                            </style>
                        </head>
                        <link rel="stylesheet" type="text/css" href="df_style.css"/>
                        <body>
                            
                            <div class="container">
                                <h1 class="sep">Model Report</h1>
                                <br>
                                <h2>Validation Table</h1>
                                <h3>Resume</h3>
                                    {table1}
                                    <br>
                                    {table2}
                                <center>
                                <br>
                                <br>
                                <h2 class="sep1">Graphs</h2>
                                <img src="{img}"/>
                                </center>
                            </div>
                        </body>
                        </html>.
                        '''
        # OUTPUT AN HTML FILE
        with open(path_output+'.html', 'w') as f:
            f.write(html_string.format(table1=resumen_print.to_html(classes='mystyle', sparsify=False, index=False, index_names=False, justify='center'),
                                    table2=ks_print.reset_index().to_html(classes='mystyle', sparsify=False, index=False, index_names=False, justify='center'),
                                    img=name_plot+'.png'))
        fig.clear()
        plt.close(fig)
            
            
    else:
        if plots:
            # graphs
            y_pred = 1 - df[[prob]]
            y_pred['prob_0'] = 1 - y_pred[prob]
            y_pred = y_pred.to_numpy()
            plt.style.use("seaborn")
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(21, 4))
            plot_roc(df[target], y_pred, ax=ax1, plot_micro=False, plot_macro=False, classes_to_plot=1,
                    text_fontsize='small')
            plot_cumulative_gain(df[target], y_pred, ax=ax2, text_fontsize='small')
            plot_ks_statistic(df[target], y_pred, ax=ax3, text_fontsize='small')
            plot_lift_curve(df[target], y_pred, ax=ax4, text_fontsize='small')
            plot_precision_recall(df[target], y_pred, ax=ax5, plot_micro=False, classes_to_plot=1, text_fontsize='small')
            ax3.legend(loc='upper left', fontsize='small')
            ax4.legend(loc='upper right', fontsize='small')
            plt.subplots_adjust(wspace=0.2)
            plt.tight_layout()
            plt.show()        
        return {'resume_sc': resumen_print,
                 'score_card': ks_print}

