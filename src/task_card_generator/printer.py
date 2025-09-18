"""Thermal printer functionality."""

from escpos.printer import Usb


def print_to_thermal_printer(image_path):
    """Print image to thermal printer."""

    try:
        # Initialize USB printer (replace with your printer's vendor/product IDs)
        printer = Usb(0x0fe6, 0x811e)  # TODO: Set correct vendor/product IDs

        # Print the image
        printer.image(image_path, impl="bitImageColumn", center=True)

        # Cut the paper
        printer.cut()

        print("Successfully printed to thermal printer!")

    except Exception as e:
        print(f"ERROR printing to thermal printer: {str(e)}")
