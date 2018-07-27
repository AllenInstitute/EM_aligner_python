import json
import subprocess
import sys
import time

import argschema
from argschema import ArgSchemaParser, ArgSchema
import matplotlib
matplotlib.use('Agg')  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from pymongo import MongoClient  # noqa: E402
import renderapi  # noqa: E402

from .. EMaligner import make_dbconnection, get_matches  # noqa: E402
from .. EM_aligner_python_schema import *  # noqa: E402


class CheckPointMatches(argschema.ArgSchemaParser):
    default_schema = EMA_PlotSchema

    def run(self, readpm_only=False):
        if self.args['z1'] > self.args['z2']:
            tmp = self.args['z1']
            self.args['z1'] = self.args['z2']
            self.args['z2'] = tmp
        self.make_plot(self.args['z1'], self.args['z2'],
                       self.args['input_stack'], self.args['pointmatch'],
                       self.args['plot'])

    def specs(self, z, stack):
        # force render, so we can read the bbox
        stack['db_interface'] = 'render'
        dbconnection = make_dbconnection(stack)
        xc = []
        yc = []
        tid = []
        tspecs = renderapi.tilespec.get_tile_specs_from_z(
            stack['name'], float(z), render=dbconnection)
        for k in np.arange(len(tspecs)):
            xc.append(0.5*tspecs[k].bbox[0]+tspecs[k].bbox[2])
            yc.append(0.5*tspecs[k].bbox[1]+tspecs[k].bbox[3])
            tid.append(tspecs[k].tileId)
        return np.array(xc), np.array(yc), np.array(tid)

    def make_plot(self, z1, z2, stack, collection, plot):
        cmap = plt.cm.plasma_r
        x1, y1, id1 = self.specs(z1, stack)
        x2, y2, id2 = self.specs(z2, stack)
        dbconnection = make_dbconnection(collection)

        # use mongo to get the point matches
        stack['db_interface'] = 'render'
        render = renderapi.connect(**stack)
        iId = renderapi.stack.get_sectionId_for_z(
            stack['name'], z1, render=render)
        jId = renderapi.stack.get_sectionId_for_z(
            stack['name'], z2, render=render)
        self.pm = get_matches(iId, jId, collection, dbconnection)
        print('%d tile pairs for z1,z2=%d,%d in collection %s__%s' %
              (len(self.pm), z1, z2, collection['owner'], collection['name']))
        if not plot:
            return

        # will hold coordinates of line segments (between tile pairs)
        lclist = []
        clist = []  # will hold number of point match pairs, used as color

        # tiny line to make sure zero is in there for consistent color range
        tmp = []
        tmp.append((0, 0))
        tmp.append((0, 0.1))
        lclist.append(tmp)
        clist.append(0)
        # tiny line to make sure max is in there for consistent color range
        tmp = []
        tmp.append((0.1, 0.1))
        tmp.append((0, 0.1))
        lclist.append(tmp)
        clist.append(self.args['threshold'])
        # if z1!=z2:
        #   clist.append(500) #limit was set at 500 for cross-section matches
        # else:
        #   clist.append(200) #limit was set at 200 for within section matches

        ntp = 0
        if len(self.pm) != 0:  # only plot if there are matches
            xmin = 1e9
            xmax = -1e9
            ymin = 1e9
            ymax = -1e9
            for k in np.arange(len(self.pm)):
                # find the tilespecs
                k1 = np.argwhere(id1 == self.pm[k]['pId']).flatten()
                k2 = np.argwhere(id2 == self.pm[k]['qId']).flatten()
                if (k1.size != 0) & (k2.size != 0):
                    ntp += 1
                    k1 = k1[0]
                    k2 = k2[0]
                    tmp = []
                    tmp.append((x1[k1], y1[k1]))
                    tmp.append((x2[k2], y2[k2]))
                    lclist.append(tmp)
                    clist.append(len(self.pm[k]['matches']['q'][0]))
                    for ix in [x1[k1], x2[k2]]:
                        if ix.min() < xmin:
                            xmin = ix.min()
                        if ix.max() > xmax:
                            xmax = ix.max()
                    for iy in [y1[k1], y2[k2]]:
                        if iy.min() < ymin:
                            ymin = iy.min()
                        if iy.max() > ymax:
                            ymax = iy.max()
            print('%d tile pairs match stack %s__%s__%s' %
                  (ntp, stack['owner'], stack['project'], stack['name']))

            # plot the line segments all at once for speed:
            # https://matplotlib.org/examples/pylab_examples/line_collection2.html
            fig = plt.figure(1, figsize=(11.69, 8.27))
            fig.clf()
            ax = fig.add_subplot(111)
            border = 4000
            ax.set_xlim(xmin-border, xmax+border)
            ax.set_ylim(ymin-border, ymax+border)

            LC = LineCollection(lclist, cmap=cmap)
            LC.set_array(np.array(clist))
            ax.add_collection(LC)
            fig = plt.gcf()
            ax.set_aspect('equal')
            # gray background, easier to see yellow
            ax.patch.set_color([0.5, 0.5, 0.5])
            ax.set_title('%s %s\n%d tile pairs %d point pairs' %
                         (z1, z2, len(self.pm), sum(clist[2:])))
            ax.invert_yaxis()
            fig.colorbar(LC)
            plt.draw()
            fname = '%s/%s_%d_%d.pdf' % (self.args['plot_dir'],
                                         collection['name'], z1, z2)
            pdf = PdfPages(fname)
            pdf.savefig(fig)  # save the figure as a pdf page
            pdf.close()
            plt.ion()
            plt.show()
            print('wrote %s' % fname)
            self.outputname = fname


if __name__ == '__main__':
    t0 = time.time()
    mod = CheckPointMatches(schema_type=EMA_PlotSchema)
    mod.run()
    print('total time: %0.1f' % (time.time()-t0))
