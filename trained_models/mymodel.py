from PIL import Image as im
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import webbrowser 

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def load_model_image():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(select_largest=False, post_process=False, device=device).eval()
    model = InceptionResnetV1(pretrained="vggface2", classify=True, num_classes=1, device=device).eval()

    checkpoint = torch.load(r"trained_models\resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return mtcnn, model

def detect_image(input_path:str, mtcnn, model):
    input_image = im.open(input_path).convert('RGB')
    face = mtcnn(input_image)
    if face is None:
        return 'No face detected'

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    face = face.to(DEVICE)
    face = face.to(torch.float32)
    face = face / 255.0

    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "Real" if output.item() < 0.1 else "Fake"

    return prediction