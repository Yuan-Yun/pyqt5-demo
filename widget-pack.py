import pandas as pd
from datetime import *
import numpy as np
from scipy.optimize import minimize
from dateutil.relativedelta import relativedelta
import os

# os.chdir('C:/Users/ryuan/Desktop/风险平价')
# df_close = pd.read_excel('to袁韵.xls', index_col='日期')  # 收盘价序列
# df_ret = df_close.iloc[1:].reset_index().iloc[:, 1:] / df_close.iloc[0:-1].reset_index().iloc[:, 1:] - 1  # 收益率序列
# df_ret.index = df_close.index[1:]
"""=======================================================================
   ---------------------------  资产配置  ----------------------------------
   ======================================================================="""


def assets_allocation(rate_of_return, start_date, end_date, method):
    return_mat = rate_of_return.loc[start_date:end_date, :]
    cov_mat = return_mat.cov() * 240  # 一年按240天计算
    omega = np.matrix(cov_mat)  # 协方差矩阵

    # 定义目标函数
    def min_variance_fun(x):
        return np.matrix(x) * omega * np.matrix(x).T

    def risk_parity_fun(x):
        tmp = (omega * np.matrix(x).T).A1
        risk = x * tmp
        delta_risk = [sum((i - risk) ** 2) for i in risk]
        return sum(delta_risk)

    def max_diversification_fun(x):
        den = x * omega.diagonal().T
        import numpy as np
        num = np.sqrt(np.matrix(x) * omega * np.matrix(x).T)
        return num / den

    # 初始值 约束条件
    x0 = np.ones(omega.shape[0]) / omega.shape[0]
    bnds = tuple((0, None) for x in x0)
    cons = ({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    options = {'disp': False, 'maxiter': 1000, 'ftol': 1e-20}

    if method == '最小方差':
        res = minimize(min_variance_fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == '风险平价':
        res = minimize(risk_parity_fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == '最大分散度':
        res = minimize(max_diversification_fun, x0, bounds=bnds, constraints=cons, method='SLSQP', options=options)
    elif method == '等权重':
        return pd.Series(index=cov_mat.index, data=1.0 / cov_mat.shape[0])

    if res['success'] == False:
        pass
    wts = pd.Series(index=cov_mat.index, data=res['x'])
    return wts


def change_date(end_day, period, direction='back'):
    end_day = datetime.strptime(end_day, '%Y-%m-%d')
    if direction == 'back':
        x = (end_day - relativedelta(months=+period)).strftime('%Y-%m-%d')
    elif direction == 'forward':
        x = (end_day + relativedelta(months=+period)).strftime('%Y-%m-%d')
    else:
        raise ValueError('direction should be back/forward')
    return x


def nav_strategy(rate_of_return, asset_allocation):
    if (not isinstance(rate_of_return, pd.DataFrame)) | (not isinstance(asset_allocation, pd.DataFrame)):
        raise TypeError('data must be <class \'pandas.core.frame.DataFrame\'>')
    if not isinstance(rate_of_return.index[0], str):
        rate_of_return.index = [str(i)[0:10] for i in rate_of_return.index]
    date_index = asset_allocation.index
    date_index = list(date_index) + [rate_of_return.index[-1]]
    strategy = pd.DataFrame(index=rate_of_return.loc[date_index[0]:].index, columns=['ret'])
    index0 = date_index[0]
    for i in date_index[1:]:
        index1 = i
        strategy.loc[index0:index1, 'ret'] = np.sum(rate_of_return[index0:index1] * asset_allocation.loc[index0], axis=1)
    strategy['nav'] = (1 + strategy['ret']).cumprod()
    bench = ((rate_of_return.iloc[(list(rate_of_return.index).index(date_index[0]) - 1):] + 1).cumprod())
    bench_strategy = pd.concat(
        [bench, pd.DataFrame({'strategy_NAV': [1.] + list(strategy['nav'])}, index=rate_of_return.index[(list(rate_of_return.index).index(date_index[0]) - 1):])], axis=1)
    return bench_strategy


def model_ana(nav):
    x = pd.DataFrame(index=nav.columns, columns=['ret_yearly', 'vol_yearly', 'draw', 'sharp'])
    x['ret_yearly'] = nav.iloc[-1] ** (240 / len(nav)) - 1  # 一年用240天
    x['vol_yearly'] = np.std(nav.iloc[1:].values / nav.iloc[0:-1].values, axis=0) * np.sqrt(240)  # 一年用240天
    x['sharp'] = x['ret_yearly'] / x['vol_yearly']

    def cal_maxdrawdown(data):
        if isinstance(data, list):
            data = np.array(data)
        if isinstance(data, pd.Series):
            data = data.values

        def get_mdd(values):  # values为np.array的净值曲线，初始资金为1
            dd = [values[i:].min() / values[i] - 1 for i in range(len(values))]
            return abs(min(dd))

        if not isinstance(data, pd.DataFrame):
            return get_mdd(data)
        else:
            return data.apply(get_mdd)

    x['draw'] = cal_maxdrawdown(nav)
    return x


def all_to_nav(rate_of_return, review_period, std_calcu_period, assets_allocation_method, end_day):
    end_day1 = end_day
    df_asset_allocate = pd.DataFrame(columns=rate_of_return.columns)
    while datetime.strptime(end_day1, '%Y-%m-%d') < datetime.strptime(str(rate_of_return.index[-1]), '%Y-%m-%d %H:%M:%S'):
        df_asset_allocate.loc[end_day1] = assets_allocation(rate_of_return, change_date(end_day1, std_calcu_period), end_day1, assets_allocation_method).values
        end_day1 = change_date(end_day1, review_period, 'forward')
    return df_asset_allocate


"""=======================================================================
   -----------------------------  界面  -----------------------------------
   ======================================================================="""

import sys, os, random

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class AppForm(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.setWindowTitle('资产配置')

        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()

        self.on_draw()

    def save_plot(self):
        file_choices = "PNG (*.png)|*.png"

        path = QFileDialog.getSaveFileName(self,
                                           'Save file', '',
                                           file_choices)
        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)

    def on_about(self):
        msg = """ A demo of using PyQt with matplotlib:

         * Use the matplotlib navigation bar
         * Add values to the text box and press Enter (or click "Draw")
         * Show or hide the grid
         * Drag the slider to modify the width of the bars
         * Save the plot to a file using the File menu
         * Click on a bar to receive an informative message
        """
        QMessageBox.about(self, "About the demo", msg.strip())

    def on_pick(self, event):
        # The event received here is of the type
        # matplotlib.backend_bases.PickEvent
        #
        # It carries lots of information, of which we're using
        # only a small amount here.
        #
        box_points = event.artist.get_bbox().get_points()
        msg = "You've clicked on a bar with coords:\n %s" % box_points

        QMessageBox.information(self, "Click!", msg)

    def on_draw(self):
        """ Redraws the figure
        """
        # str = unicode(self.textbox.text())
        # self.data = list(map(int, self.textbox.text().split()))
        os.chdir('C:/Users/ryuan/Desktop/风险平价')
        df_close = pd.read_excel('to袁韵.xls', index_col='日期')  # 收盘价序列
        df_ret = df_close.iloc[1:].reset_index().iloc[:, 1:] / df_close.iloc[0:-1].reset_index().iloc[:, 1:] - 1  # 收益率序列
        df_ret.index = df_close.index[1:]
        df_asset = all_to_nav(df_ret, review_period=self.ReviewPeriodSpinBox.value(), std_calcu_period=self.StdPeriodSpinBox.value(),
                              end_day=self.StartDateEdit.date().toString('yyyy-MM-dd'), assets_allocation_method=self.ModelChooseComboBox.currentText())
        df_result = nav_strategy(df_ret, df_asset)

        x = range(len(df_result))
        y = df_result['strategy_NAV']

        # clear the axes and redraw the plot anew
        #
        self.axes.clear()
        self.axes.grid(self.grid_cb.isChecked())

        # self.axes.bar(
        #     left=x,
        #     height=self.data,
        #     width=self.slider.value() / 100.0,
        #     align='center',
        #     alpha=0.44,
        #     picker=5)
        self.axes.plot(x, y)

        self.canvas.draw()

    def create_main_frame(self):
        self.main_frame = QWidget()

        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 100
        self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

        # Since we have only one plot, we can use add_axes
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)

        # Bind the 'pick' event for clicking on one of the bars
        #
        self.canvas.mpl_connect('pick_event', self.on_pick)

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self.main_frame)

        # Other GUI controls
        #
        self.ReviewPeriodLabel = QLabel('回顾间隔月数:')
        self.ReviewPeriodSpinBox = QSpinBox()
        self.ReviewPeriodSpinBox.setMinimum(1)
        self.ReviewPeriodSpinBox.setMaximum(12)
        self.ReviewPeriodSpinBox.setSingleStep(1)
        self.ReviewPeriodSpinBox.setProperty("value", 3)

        self.draw_button = QPushButton("画图")
        self.draw_button.clicked.connect(self.on_draw)

        self.grid_cb = QCheckBox("网格线")
        self.grid_cb.setChecked(False)
        self.grid_cb.stateChanged.connect(self.on_draw)  # int

        self.StdPeriodLabel = QLabel('风险计算时长:')
        self.StdPeriodSpinBox = QSpinBox()
        self.StdPeriodSpinBox.setMinimum(6)
        self.StdPeriodSpinBox.setMaximum(36)
        self.StdPeriodSpinBox.setSingleStep(6)
        self.StdPeriodSpinBox.setProperty("value", 12)

        self.startdate_label = QLabel("回测起始日期:")
        self.StartDateEdit = QDateEdit()
        self.StartDateEdit.setDateTime(QDateTime(QDate(2014, 12, 31), QTime(0, 0, 0)))
        self.StartDateEdit.setMaximumDateTime(QDateTime(QDate(2016, 12, 30), QTime(23, 59, 59)))
        self.StartDateEdit.setMinimumDateTime(QDateTime(QDate(2014, 12, 31), QTime(0, 0, 0)))
        self.StartDateEdit.setDisplayFormat("yyyy-MM-dd")

        self.modelChoose_label = QLabel('模型选择:')
        self.SaveCheckBox = QCheckBox()
        self.ModelChooseComboBox = QComboBox()
        self.ModelChooseComboBox.addItem("等权重")
        self.ModelChooseComboBox.addItem("风险平价")
        self.ModelChooseComboBox.addItem("最小方差")
        self.ModelChooseComboBox.addItem("最大分散度")

        hbox = QGridLayout()
        hbox.setSpacing(25)
        hbox.addWidget(self.ReviewPeriodLabel, 0, 0, 1, 1)
        hbox.addWidget(self.ReviewPeriodSpinBox, 0, 1, 1, 1)
        hbox.addWidget(self.StdPeriodLabel, 0, 2, 1, 1)
        hbox.addWidget(self.StdPeriodSpinBox, 0, 3, 1, 1)
        hbox.addWidget(self.startdate_label, 0, 4, 1, 1)
        hbox.addWidget(self.StartDateEdit, 0, 5, 1, 1)
        hbox.addWidget(self.modelChoose_label, 1, 0, 1, 1)
        hbox.addWidget(self.ModelChooseComboBox, 1, 1, 1, 2)
        hbox.addWidget(self.grid_cb, 1, 4, 1, 1)
        hbox.addWidget(self.draw_button, 1, 5, 1, 1)

        # #
        # # Layout with box sizers
        # #
        # hbox = QHBoxLayout()
        #
        # for w in [self.textbox, self.draw_button, self.grid_cb,
        #           slider_label, self.slider]:
        #     hbox.addWidget(w)
        #     hbox.setAlignment(w, Qt.AlignVCenter)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.canvas)
        vbox.addWidget(self.mpl_toolbar)



        self.main_frame.setLayout(vbox)
        self.setCentralWidget(self.main_frame)
        # icon
        self.setWindowIcon(QIcon('C:/Users/ryuan/Pictures/Saved Pictures/strategy_128px_1208206_easyicon.net.ico'))
        # background
        palette1 = QPalette()
        palette1.setColor(self.backgroundRole(), QColor('white'))  # 设置背景颜色
        self.setPalette(palette1)
        # font
        w = [self.ReviewPeriodLabel,self.StdPeriodLabel,self.startdate_label,self.modelChoose_label,self.grid_cb,self.draw_button]
        for i in w:
            i.setStyleSheet("font: 75 10pt \"微软雅黑\"")
        w = [self.ReviewPeriodSpinBox, self.StdPeriodSpinBox, self.StartDateEdit]
        for i in w:
            i.setStyleSheet("font: 10pt \"Times New Roman\"")
        self.ModelChooseComboBox.setStyleSheet("font: 10pt \"微软雅黑\"")
        self.draw_button.setStyleSheet("border:2px white;border-radius:10px;padding:2px 4px;background-color: steelblue; color: white; font: 75 12pt \"微软雅黑\";")

    def create_status_bar(self):
        self.status_text = QLabel("Powered by PyQT5")
        self.statusBar().addWidget(self.status_text, 1)

    def create_menu(self):
        self.file_menu = self.menuBar().addMenu("&File")

        load_file_action = self.create_action("&Save plot",
                                              shortcut="Ctrl+S", slot=self.save_plot,
                                              tip="Save the plot")
        quit_action = self.create_action("&Quit", slot=self.close,
                                         shortcut="Ctrl+Q", tip="Close the application")

        self.add_actions(self.file_menu,
                         (load_file_action, None, quit_action))

        self.help_menu = self.menuBar().addMenu("&Help")
        about_action = self.create_action("&About",
                                          shortcut='F1', slot=self.on_about,
                                          tip='About the demo')

        self.add_actions(self.help_menu, (about_action,))

    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def create_action(self, text, slot=None, shortcut=None,
                      icon=None, tip=None, checkable=False,
                      signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action


def main():
    app = QApplication(sys.argv)
    form = AppForm()
    form.show()
    app.exec_()


if __name__ == "__main__":
    main()
