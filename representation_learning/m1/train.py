import torch
import numpy as np


def predict(batch, enc, dec_1, dec_2, dec_3, fc, loss_fn, device):
    with torch.no_grad():
        batch = batch.to(device)
        decoder_1_h, decoder_1_c, decoder_2_h, decoder_2_c, decoder_3_h, decoder_3_c = enc(
            batch)
        decoder_1_in = torch.zeros((batch.shape[1:])).to(device)
        outputs = []
        loss = 0
        for i in range(batch.shape[0]):
            decoder_1_h, decoder_1_c = dec_1(
                decoder_1_in, (decoder_1_h, decoder_1_c))
            decoder_2_h, decoder_2_c = dec_2(
                decoder_1_h, (decoder_2_h, decoder_2_c))
            decoder_3_h, decoder_3_c = dec_3(
                decoder_2_h, (decoder_3_h, decoder_3_c))
            output = fc(decoder_3_h)
            decoder_1_in = output
            loss += loss_fn(output, batch[i]).item()
            outputs.append(output)
        outputs = torch.stack(outputs)
    return outputs, loss


def fit(train_loader, val_loader, enc, dec_1, dec_2, dec_3, fc, loss_fn, optimizer, lr_sched, lr_max_strikes, n_epochs, model_save_path, device):
    train_losses, val_losses = [], []
    best_val_loss = torch.tensor(float('inf'))
    lr_strike = 0
    for epoch in range(1, n_epochs + 1):
        enc.train()
        dec_1.train()
        dec_2.train()
        dec_3.train()
        fc.train()
        train_loss = []
        for batch in train_loader:
            batch = batch.to(device)

            decoder_1_h, decoder_1_c, decoder_2_h, decoder_2_c, decoder_3_h, decoder_3_c = enc(
                batch)
            decoder_1_h = decoder_1_h.squeeze()
            decoder_1_c = decoder_1_c.squeeze()
            decoder_2_h = decoder_2_h.squeeze()
            decoder_2_c = decoder_2_c.squeeze()
            decoder_3_h = decoder_3_h.squeeze()
            decoder_3_c = decoder_3_c.squeeze()

            decoder_1_in = torch.zeros((batch.shape[1:])).to(device)

            outputs = []
            loss = torch.tensor(0, dtype=torch.float32).to(device)

            for i in range(batch.shape[0]):
                decoder_1_h, decoder_1_c = dec_1(
                    decoder_1_in, (decoder_1_h, decoder_1_c))
                decoder_2_h, decoder_2_c = dec_2(
                    decoder_1_h, (decoder_2_h, decoder_2_c))
                decoder_3_h, decoder_3_c = dec_3(
                    decoder_2_h, (decoder_3_h, decoder_3_c))
                output = fc(decoder_3_h)

                decoder_1_in = output

                loss += loss_fn(output, batch[i])
                outputs.append(output)

            outputs = torch.stack(outputs)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_strike += 1

        val_loss = []
        enc.eval()
        dec_1.eval()
        dec_2.eval()
        dec_3.eval()
        fc.eval()
        for batch in val_loader:
            _, val_batch_loss = predict(
                batch, enc, dec_1, dec_2, dec_3, fc, loss_fn, device)
            val_loss.append(val_batch_loss)

        train_loss = np.mean(train_loss)
        train_losses.append(train_loss)
        val_loss = np.mean(val_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            print('best model', val_loss)
            torch.save(enc.state_dict(), f'{model_save_path}/{epoch}_enc.pth')
            torch.save(dec_1.state_dict(), f'{model_save_path}/{epoch}_dec_1.pth')
            torch.save(dec_2.state_dict(), f'{model_save_path}/{epoch}_dec_2.pth')
            torch.save(dec_3.state_dict(), f'{model_save_path}/{epoch}_dec_3.pth')
            torch.save(fc.state_dict(), f'{model_save_path}/{epoch}_fc.pth')
            best_val_loss = val_loss
            lr_strike = 0

        if lr_strike > lr_max_strikes:
            lr_sched.step()
            print('reducing lr to', optimizer.param_groups[0]['lr'])
            lr_strike = 0

        print(
            f"epoch {epoch:>3}\ttraining loss: {train_loss:0.5f}\tval loss: {val_loss:0.5f}")

    return train_losses, val_losses, (enc, dec_1, dec_2, dec_3, fc)
