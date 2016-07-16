# -*- coding: utf-8 -*-
# Copyright (c) 2016 Ryusuke Sasaki
# Released under the MIT license
# http://opensource.org/licenses/mit-license.php
#
u"""
Green Coordinates によるケージデフォーマーのテスト実装。

Python API 2.0 で MPxDeformerNode がサポートされていなかった為、旧 API で実装した。
習作のため、いろいろと実用レベルではない。

参考:
http://www.wisdom.weizmann.ac.il/~ylipman/GC/gc_techrep.pdf
"""
__author__ = 'ryusas'
__version__ = '1.0.0'

#==================================================================================
_TYPE_ID = 0x00070003  # もし何かと衝突するなら書き換えること。(Free: 0x00000000 ～ 0x0007ffff)

import sys
from math import sin, cos, acos, atan2, sqrt, log, pi as _PI
#from pprint import pprint as _pprt

import maya.OpenMaya as api1
import maya.OpenMayaMPx as api1px
from maya.OpenMaya import MFnMesh, MFloatPointArray, MIntArray, MPoint

import maya.api.OpenMaya
from maya.api.OpenMaya import (
    MVector as api2_MVector,
    MPoint as api2_MPoint,
    MMatrix as api2_MMatrix,
)


#==================================================================================
if api1.MGlobal_apiVersion() < 201600:
    _deformerAttr = lambda name: getattr(api1px.cvar, 'MPxDeformerNode_' + name)
else:
    _deformerAttr = lambda name: getattr(api1px.cvar, 'MPxGeometryFilter_' + name)

_NONE_LIST = [None]
_ZERO_LIST = [0.]
_RANGE3 = (0, 1, 2)
_ZERO_VEC = api2_MVector.kZeroVector
_SQRT_8 = sqrt(8.)

_MIN_FLOAT = sys.float_info.min
_fourPI = 4. * _PI

_sign = lambda x: -1. if x < 0. else 1.

_dot_VV_V = lambda vv, v: vv[0] * v[0] + vv[1] * v[1] + vv[2] * v[2]

_EPSILON = 1.e-10
_denom = lambda x: _sign(x) * max(abs(x), _EPSILON)


def _m1to2(m):
    return api2_MMatrix([
        m(0, 0), m(0, 1), m(0, 2), m(0, 3),
        m(1, 0), m(1, 1), m(1, 2), m(1, 3),
        m(2, 0), m(2, 1), m(2, 2), m(2, 3),
        m(3, 0), m(3, 1), m(3, 2), m(3, 3),
    ])


#------------------------------------------------------------------------------
class Cage(object):
    u"""
    ケージクラス。
    Maya のメッシュデータから情報を取得し構築する。
    構築後は Maya のデータは持たない。
    baseCage の場合は、ゼロから生成。
    inCage の場合は baseInfo を与え、それを元に座標値などが取得される。
    """
    def __init__(self, meshData, baseInfo=None):
        self._meshData = meshData
        fn = MFnMesh(meshData)

        # 頂点座標リストを生成。各要素は API 2.0 の MVector とする。
        pntArr = MFloatPointArray()
        fn.getPoints(pntArr)
        nVrts = pntArr.length()
        if baseInfo and baseInfo[0] != nVrts:
            # ベースケージと頂点数が合わない場合は無効とする。
            self._vertices = None
            return
        vertices = _NONE_LIST * nVrts
        for i in xrange(nVrts):
            p = pntArr[i]
            vertices[i] = api2_MVector(p.x, p.y, p.z)
        #_pprt(vertices)
        self._vertices = vertices

        # ベースケージの場合はトライアングル情報を構築する。
        # トライアングルリストを生成。各要素は３頂点のインデクスリストとする。
        if baseInfo:
            triangles = baseInfo[1]
            self._triRestEdges = baseInfo[2]
            self._triRestAreas = baseInfo[3]
        else:
            triCounts = MIntArray()
            triVrts = MIntArray()
            fn.getTriangles(triCounts, triVrts)
            nPlys = triCounts.length()
            #print(triCounts.length(), triVrts.length())
            triangles = []
            vi = 0
            for i in xrange(nPlys):
                for j in range(triCounts[i]):
                    triangles.append([triVrts[k] for k in range(vi, vi + 3)])
                    vi += 3
            #_pprt(triangles)
        self._triangles = triangles

        # トライアングルのエッジベクトルを計算。
        self._triEdges = [(vertices[tri[1]] - vertices[tri[0]], vertices[tri[2]] - vertices[tri[1]]) for tri in triangles]

        # トライアングルの法線ベクトルを計算。
        self._triNrms = [(u ^ v).normalize() for u, v in self._triEdges]

        # ベースケージの場合はトライアングルの面積を計算。変形ケージでは不要。
        if not baseInfo:
            self._triAreas = [(u ^ v).length() * .5 for u, v in self._triEdges]

    def isValid(self):
        u"""
        変形ケージがベースケージに基づくものだったかどうか。
        """
        return bool(self._vertices)

    def baseInfo(self):
        u"""
        ベースケージの場合に、バインドデータに保存する情報を得る。
        """
        return len(self._vertices), self._triangles, self._triEdges, self._triAreas

    def computeDeform(self, phi, psi):
        u"""
        ターゲットの一つの点のデフォーム位置を計算する。
        """
        pnt = api2_MPoint()

        for p, v in zip(phi, self._vertices):
            pnt += v * p
        for p, (u, v), n, (ru, rv), ra in zip(psi, self._triEdges, self._triNrms, self._triRestEdges, self._triRestAreas):
            p *= sqrt((u * u) * (rv * rv) - 2. * (u * v) * (ru * rv) + (v * v) * (ru * ru)) / _denom(_SQRT_8 * ra)
            pnt += n * p

        #for i, v in enumerate(self._vertices):
        #    pnt += v * phi[i]
        #uvs = self._triEdges
        #ruvs = self._triRestEdges
        #ras = self._triRestAreas
        #for i, n in enumerate(self._triNrms):
        #    u, v = uvs[i]
        #    ru, rv = ruvs[i]
        #    p = psi[i] * sqrt((u * u) * (rv * rv) - 2. * (u * v) * (ru * rv) + (v * v) * (ru * ru)) / _denom(_SQRT_8 * ras[i])
        #    pnt += n * p

        return pnt

    def computeCoords(self, pvec):
        u"""
        ターゲットの一つの点の Green Coordinates を計算する。
        """
        vertices = self._vertices
        triangles = self._triangles
        triNrms = self._triNrms
        nVrts = len(vertices)
        nTris = len(triangles)

        phi = _ZERO_LIST * nVrts
        psi = _ZERO_LIST * nTris

        s = api2_MVector()
        I = api2_MVector()
        II = api2_MVector()
        N = [None, None, None]

        for i, tri in enumerate(triangles):
            nrm = triNrms[i]
            vj = [(vertices[vi] - pvec) for vi in tri]
            p = (vj[0] * nrm) * nrm

            for k in _RANGE3:
                v0 = vj[k]
                v1 = vj[(k + 1) % 3]
                s[k] = _sign(((v0 - p) ^ (v1 - p)) * nrm)
                I[k] = _gcTriInt(p, v0, v1)
                II[k] = _gcTriInt(_ZERO_VEC, v1, v0)
                N[k] = (v1 ^ v0).normalize()

            I_ = -abs(s * I)
            psi[i] -= I_
            w = I_ * nrm + _dot_VV_V(N, II)
            if w.length() >= _EPSILON:
                for k in _RANGE3:
                    vn = N[(k + 1) % 3]
                    phi[tri[k]] += (vn * w) / (vn * vj[k])

        s = sum(phi)
        if s >= _EPSILON:
            s = 1. / s
            phi = [p * s for p in phi]

        return phi, psi


def _gcTriInt(p, v1, v2):  #, eta=_ZERO_VEC):
    v2_v1 = v2 - v1
    p_v1 = p - v1
    v1_p = -p_v1
    v2_p = v2 - p

    alpha = acos(min(1., max(-1., (v2_v1 * p_v1) / (v2_v1.length() * p_v1.length()) )))
    beta = acos(min(1., max(-1., (v1_p * v2_p) / (v1_p.length() * v2_p.length()) )))
    sin_alpha = sin(alpha)
    lamda = (p_v1 * p_v1) * sin_alpha * sin_alpha

    #p_eta = p - eta
    #c = p_eta * p_eta
    c = p * p

    sqrt_c = sqrt(c)
    sqrt_lamda = sqrt(lamda)
    lamdalamda = lamda * lamda

    def _I(theta):
        S = sin(theta)
        C = cos(theta)
        SS = S * S
        one_C = 1. - C
        cC = c * C

        denom = one_C * one_C;
        res = 2. * sqrt_lamda * SS / _denom(one_C * one_C) * (1. - 2. * cC / _denom(c + cC + lamda + sqrt(lamdalamda + lamda * c * SS)))
        res = log(max(res, _MIN_FLOAT))
        res *= sqrt_lamda
        res += 2. * sqrt_c * atan2(sqrt_c * C, sqrt(lamda + SS * c))
        res *= _sign(S) / -2. 
        return res

    theta = _PI - alpha
    return -abs(_I(theta) - _I(theta - beta) - sqrt_c * beta) / _fourPI


#------------------------------------------------------------------------------
class GreenCageDeformer(api1px.MPxDeformerNode):
    type_name = 'greenCageDeformer'
    type_id = api1.MTypeId(_TYPE_ID)

    @classmethod
    def initialize(cls):
        fnTyped = api1.MFnTypedAttribute()
        fnNumeric = api1.MFnNumericAttribute()

        cls.aInCage = fnTyped.create('inCage', 'ic', api1.MFnData.kMesh)
        fnTyped.setStorable(False)
        fnTyped.setCached(False)
        cls.addAttribute(cls.aInCage)

        cls.aBaseCage = fnTyped.create('baseCage', 'bc', api1.MFnData.kMesh)
        cls.addAttribute(cls.aBaseCage)

        cls.aEnvelope = _deformerAttr('envelope')
        cls.aInput = _deformerAttr('input')
        cls.aOutputGeom = _deformerAttr('outputGeom')
        cls.attributeAffects(cls.aInCage, cls.aOutputGeom)
        cls.attributeAffects(cls.aBaseCage, cls.aOutputGeom)

    def __init__(self, *a, **ka):
        super(GreenCageDeformer, self).__init__(*a, **ka)
        self._bindDataDict = {}

    def setDependentsDirty(self, plug, affected):
        u"""
        plug が dirty になった時に呼ばれるので、
        ケージとターゲットの元形状の入力に変化があったらバインドデータを破棄する。
        """
        if plug != self.aBaseCage:
            if plug.isChild():
                plug = plug.parent()
            if plug != self.aInput:
                return
        self._bindDataDict = {}

    def _getBindData(self, block, geomIt, matrix, index):
        u"""
        ターゲットインデクスからバインド情報を得る。無ければバインドする。
        """
        bindData = self._bindDataDict.get(index)
        if bindData:
            return bindData

        nPnts = geomIt.count()
        if not nPnts:
            return

        hBase = block.inputValue(self.aBaseCage)
        if hBase.data().isNull():
            return

        #print('BIND')
        cage = Cage(hBase.asMeshTransformed())

        weights = _NONE_LIST * nPnts
        i = 0
        while not geomIt.isDone():
            p = geomIt.position()
            p *= matrix
            weights[i] = cage.computeCoords(api2_MVector(p.x, p.y, p.z))
            geomIt.next()
            i += 1
        geomIt.reset()

        bindData = cage.baseInfo(), weights
        self._bindDataDict[index] = bindData
        return bindData

    def deform(self, block, geomIt, matrix, index):
        # envelope=0 なら無視。
        # ちなみに nodeState=hasNoEffect では deform が呼ばれないからそのチェックは不要。
        envelope = block.inputValue(self.aEnvelope).asFloat()
        if not envelope:
            return

        hCage = block.inputValue(self.aInCage)
        if hCage.data().isNull():
            return

        bindData = self._getBindData(block, geomIt, matrix, index)
        if not bindData:
            return
        baseInfo, weights = bindData
        #print('DEFORM')

        cage = Cage(hCage.asMeshTransformed(), baseInfo)
        if not cage.isValid():
            return

        # 手抜きで、envelope 値とデフォーマーウェイトは無視。
        invM = _m1to2(matrix).inverse()
        i = 0
        while not geomIt.isDone():
            p = cage.computeDeform(*weights[i])
            p *= invM
            geomIt.setPosition(MPoint(p.x, p.y, p.z))
            geomIt.next()
            i += 1


#------------------------------------------------------------------------------
def _registerNode(plugin, cls, *args):
    try:
        plugin.registerNode(cls.type_name, cls.type_id, lambda: cls(), cls.initialize, *args)
    except:
        sys.stderr.write('Failed to register node: ' + cls.type_name)
        raise


def _deregisterNode(plugin, cls):
    try:
        plugin.deregisterNode(cls.type_id)
    except:
        sys.stderr.write('Failed to deregister node: ' + cls.type_name)
        raise


def initializePlugin(mobj):
    plugin = api1px.MFnPlugin(mobj, __author__, __version__, 'Any')
    _registerNode(plugin, GreenCageDeformer, api1px.MPxNode.kDeformerNode)


def uninitializePlugin(mobj):
    plugin = api1px.MFnPlugin(mobj)
    _deregisterNode(plugin, GreenCageDeformer)

