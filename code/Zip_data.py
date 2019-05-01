import os, zipfile, tarfile

def zipup_data(output_name=r'..\Reg_Data.zip', data_path = r'..\Reg_Data'):
    ziph = zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(data_path):
        for file in files:
            ziph.write(os.path.join(root,file))
    ziph.close()
    return None


def unzip_file(file_path,output_path):
    if file_path.find('.zip') != -1:
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(output_path)
        zip_ref.close()
    elif file_path.find('.tar') != -1:
        tar = tarfile.open(file_path)
        tar.extractall(output_path)
        tar.close()
    return None
class Unzip_class():
    def __init__(self, file, out_path):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        unzip_file(os.path.join(file),out_path)
        down_folder(out_path)


def down_folder(path):
    files = []
    dirs = []
    for root, dirs, files in os.walk(path):
        break
    for i, file in enumerate(files):
        print(str(i/len(files) * 100) + '% done')
        if file.find('.zip') != -1:
            print(os.path.join(root,file))
            unzip_file(os.path.join(root,file),root)
    for dir in dirs:
        down_folder(os.path.join(root,dir))


if __name__ == '__main__':
    xxx = 1