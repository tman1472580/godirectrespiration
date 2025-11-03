import argparse
import asyncio
import logging

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

logger = logging.getLogger(__name__)

def simple_callback(device: BLEDevice, advertisement_data: AdvertisementData):
    if advertisement_data and advertisement_data.local_name and advertisement_data.local_name[0:3] == 'GDX':
        logger.info("%s: %s rssi:%d", device.address, advertisement_data.local_name, advertisement_data.rssi)


async def main():
    scanner = BleakScanner(simple_callback)

    count = 1
    sleep = 2.0
    while count > 0:
        logger.info("---------------------------------------------------------------------")
        logger.info("---------------------------------------------------------------------")
        logger.info("(re)starting scanner")
        await scanner.start()
        await asyncio.sleep(sleep)
        await scanner.stop()
        count = count - 1
    logger.info("---------------------------------------------------------------------")
    logger.info("---------------------------------------------------------------------")

if __name__ == "__main__":

    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)-15s %(name)-8s %(levelname)s: %(message)s",
    )

    asyncio.run(main())