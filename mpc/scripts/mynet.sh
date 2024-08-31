# https://zhuanlan.zhihu.com/p/588714270

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

# Roger, about 1GBps
# sudo tc qdisc add dev $DEV root handle 1: tbf rate 8192mbit burst 100000 limit 10000
# Roger, about 1ms ping latency
# sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.5msec
# Flight
sudo tc qdisc add dev $DEV root handle 1: tbf rate 80gbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 0.1msec
# Alkaid
sudo tc qdisc add dev $DEV root handle 1: tbf rate 1gbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 1msec
fi

if [ "$1" == "wan" ]
then
sudo tc qdisc del dev $DEV root
# Roger, about 40MBps
# sudo tc qdisc add dev $DEV root handle 1: tbf rate 320mbit burst 100000 limit 10000
# Roger, about 70ms ping latency
# sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 35msec
# Flight
sudo tc qdisc add dev $DEV root handle 1: tbf rate 400mbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 25msec
# Alkaid
sudo tc qdisc add dev $DEV root handle 1: tbf rate 160mbit burst 100000 limit 10000
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 100msec
fi