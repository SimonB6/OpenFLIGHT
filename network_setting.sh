#!/bin/bash
######
# Taken from https://github.com/emp-toolkit/emp-readme/blob/master/scripts/throttle.sh
######

## replace DEV=lo with your card (e.g., eth0)
DEV=lo 
if [ "$1" == "del" ]
then
	sudo tc qdisc del dev $DEV root
fi

if [ "$1" == "lan" ]
then
sudo tc qdisc del dev $DEV root
## about 10Gbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 10000mbit burst 100000 limit 10000
## about 0.2ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.1msec
fi
if [ "$1" == "wan" ]
then
sudo tc qdisc del dev $DEV root
## about 50Mbps
sudo tc qdisc add dev $DEV root handle 1: tbf rate 50mbit burst 100000 limit 10000
## about 50ms ping latency
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 25msec
fi