#!/usr/bin/env python
import click
from pydr.dr import dr_single_subject

@click.command()
@click.argument('func')
@click.argument('map')
@click.argument('timecourse')
@click.argument('grp_map')
@click.argument('amplitude')
@click.argument('spectra')
@click.argument('func_smooth')

def dr_helper(*args, **kwargs):
    dr_single_subject(*args, **kwargs)

if __name__ == "__main__":
    dr_helper()
