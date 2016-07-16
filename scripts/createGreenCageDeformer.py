# -*- coding: utf-8 -*-
u"""
greenCageDeformer を作成し、ターゲットとケージをバインドするスクリプト例。

ケージ（一つのmesh）、ターゲット（任意のシェイプ、部分的なポイントや複数も可）
の順に選択してスクリプトを実行するとバインドされる。
"""
import maya.cmds as cmds


def doit(cage_tgt=None):
    if not cage_tgt:
        cage_tgt = cmds.ls(sl=True, o=True)
    cage = cage_tgt[0]
    tgt = cage_tgt[1:]

    cmds.loadPlugin('greenCageDeformer.py', qt=True)
    deformer = cmds.deformer(tgt, type='greenCageDeformer')[0]

    freezer = cmds.createNode('transformGeometry')
    cmds.connectAttr(cage + '.o', freezer + '.ig')
    cmds.connectAttr(cage + '.wm', freezer + '.txf')
    cmds.connectAttr(freezer + '.og', deformer + '.bc')
    cmds.disconnectAttr(freezer + '.og', deformer + '.bc')
    cmds.delete(freezer)

    cmds.connectAttr(cage + '.w', deformer + '.ic')
    cmds.dgeval(cmds.listConnections(deformer + '.og', s=False, d=True, sh=True, p=True))


#doit([cmds.polyCube(w=2.5, d=2.5, h=2.5)[0], cmds.polySphere()[0]])
doit()

