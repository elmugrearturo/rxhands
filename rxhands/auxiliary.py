import cv2
import math

def show_img(img, win_title=None):
    if not win_title:
        win_title = "%dx%d" % img.shape[:2]

    assert isinstance(win_title, str)

    img = img.astype('uint8')
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.imshow(win_title, img)
    cv2.resizeWindow(win_title, 1024, 768)

    while cv2.getWindowProperty(win_title,
                                cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKeyEx(1000) == 27:
            cv2.destroyWindow(win_title)
            break

def load_gray_img(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def save_img(img, output_path="../output.png"):
    if not output_path.endswith(".png"):
        output_path += ".png"
    cv2.imwrite(output_path, img)
    return True

def divide_image(img, divisions):
    h, w = img.shape
    patches = []

    # Calculate step size
    step = int(math.sqrt(divisions))
    h_step = int(h / step)
    w_step = int(w / step)
    
    # Cut patches
    last_i = 0
    i = h_step
    for count_i in range(step):
        last_j = 0
        j = w_step
        for count_j in range(step):
            patches.append(img[last_i:i, last_j:j])
            last_j = j
            if count_j == step - 2 :
                j = w
            else:
                j += w_step
        last_i = i
        if count_i == step - 2 :
            i = h
        else:
            i += h_step

    assert len(patches) == divisions
    return patches

# Método para contar componentes conexas
def count_labels(labelled_img):
    return [(label, (labelled_img == label).sum()) for label in np.unique(labelled_img) if label != 0]

# Métodos auxiliares para registrar equivalencias (OPCIONAL)
def registrar_equivalencias(valores, equivalencias):
    valores = set(valores)
    found = []
    new_equivalencias = []
    for i in range(len(equivalencias)):
        grupo = equivalencias[i]
        for valor in list(valores):
            if valor in grupo:
                if i not in found:
                    found.append(i)
    if len(found) > 0:
        for i in range(len(equivalencias)):
            if i in found:
                valores = valores.union(equivalencias[i])
            else:
                new_equivalencias.append(equivalencias[i])
    else:
        new_equivalencias = equivalencias
    new_equivalencias.append(valores)
    return new_equivalencias

def construir_tabla(equivalencias):
    tabla = {}
    for group in equivalencias:
        clase_real = min(group)
        for value in list(group):
            tabla[value] = clase_real
    return tabla

def find_connected_components(img, B):
    # receives binary image and a 1's kernel (4/8 connectivity)
    equivalencias = []
    etiqueta_actual = 2
    img_labels = np.copy(img).astype("int64")
    a = int((B.shape[0] - 1)/2)
    b = int((B.shape[1] - 1)/2)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] == 1:
                patch = np.zeros(B.shape)
                for k in range(-a, a+1):
                    for l in range(-b, b+1):
                        try:
                            patch[k+a, l+b] = img_labels[i+k, j+l]
                        except:
                            pass
                patch = patch * B
                if True in (patch > 1):
                    label = int(patch[patch > 1].min())
                    merge_candidates = set(patch[patch > 1].flatten().astype("int64"))
                    img_labels[i, j] = label
                    if len(merge_candidates) > 1:
                        equivalencias = registrar_equivalencias(
                                merge_candidates,
                                equivalencias)
                else:
                    img_labels[i, j] = etiqueta_actual
                    etiqueta_actual += 1

    tabla_equivalencias = construir_tabla(equivalencias)
    img_labels = img_labels.flatten()
    for i in range(img_labels.shape[0]):
        try:
            img_labels[i] = tabla_equivalencias[img_labels[i]]
        except:
            pass
    img_labels[img_labels > 0] -= 1
    img_labels = img_labels.reshape(img.shape)
    return img_labels
