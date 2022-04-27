from models.PCP_model import build
import argparse

def build_model(args, training=False):
    return build(args, training)
